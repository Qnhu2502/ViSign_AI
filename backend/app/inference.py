import importlib.util
import json
import logging
import os
import threading
from pathlib import Path

import cv2
import joblib
import numpy as np

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class TemporalAttentionFactory:
    @staticmethod
    def build():
        import tensorflow as tf
        from tensorflow.keras import layers

        @tf.keras.utils.register_keras_serializable(package="ViSignV3")
        class TemporalAttention(layers.Layer):
            def __init__(self, units=64, **kwargs):
                super().__init__(**kwargs)
                self.units = units
                self.dense = layers.Dense(units, activation="tanh")
                self.score = layers.Dense(1)

            def build(self, input_shape):
                self.dense.build(input_shape)
                self.score.build((*input_shape[:-1], self.units))
                super().build(input_shape)

            def call(self, inputs):
                weights = tf.nn.softmax(self.score(self.dense(inputs)), axis=1)
                return tf.reduce_sum(inputs * weights, axis=1)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[-1])

            def get_config(self):
                return {**super().get_config(), "units": self.units}

        return TemporalAttention


class Predictor:
    def __init__(self):
        self.lock = threading.RLock()
        self.ready = False
        self.error = None
        self.model = None
        self.detectors = None
        self.V = None
        self.labels = []
        self.config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        self.metrics = self.config.get("test_metrics", {})
        self.labels = []
        self._load_metadata()

    def _load_metadata(self):
        # Only load pickle files bundled with the trusted user's trained model.
        self.encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
        self.labels = [str(x) for x in self.encoder.classes_]

    def load(self):
        with self.lock:
            if self.ready:
                return
            try:
                import tensorflow as tf
                # visign_features.py defaults to Colab's /content/mp_models; keep MediaPipe .task files in
                # backend/mp_models unless VISIGN_MODEL_DIR is set (e.g. in backend/.env).
                os.environ.setdefault("VISIGN_MODEL_DIR", str(MODEL_DIR.parent / "mp_models"))
                spec = importlib.util.spec_from_file_location("visign_features_v5", MODEL_DIR / "visign_features.py")
                V = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(V)
                self.V = V
                self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
                self.model = tf.keras.models.load_model(
                    str(MODEL_DIR / "best_model.keras"),
                    custom_objects={"TemporalAttention": TemporalAttentionFactory.build()},
                    compile=False,
                )
                if tuple(self.model.input_shape[-2:]) != (30, 204):
                    raise ValueError(f"Model input mismatch: {self.model.input_shape}")
                if int(self.model.output_shape[-1]) != len(self.labels):
                    raise ValueError("Model output and label count mismatch")
                self.ready = True
                self.error = None
                logger.info("V5 loaded: %d labels", len(self.labels))
            except Exception as exc:
                self.error = str(exc)
                logger.exception("Could not load V5 model")
                raise

    def _predict_raw(self, raw):
        if not np.any(raw[:, :126]):
            raise ValueError("Không phát hiện được bàn tay trong video")
        data = self.V.apply_scaler(raw[None, ...], self.scaler)
        probabilities = self.model.predict(data, verbose=0)[0]
        order = np.argsort(probabilities)[::-1][:5]
        return [{"gloss": self.labels[int(i)], "confidence": round(float(probabilities[i]), 6)} for i in order]

    def _get_detectors(self):
        """Create the MediaPipe hand/pose/face landmarkers once and reuse them.
        They run in IMAGE mode (no state between frames), so reuse is safe; callers must hold self.lock."""
        if self.detectors is None:
            self.detectors = self.V.make_detectors(False)
        return self.detectors

    def predict_frames(self, frames):
        """frames are BGR images, sampled evenly; model inference is serialized."""
        top5, stats = self.predict_frames_with_stats(frames)
        if top5 is None:
            raise ValueError("Không phát hiện được bàn tay trong video")
        return top5

    @staticmethod
    def landmark_stats(V, raw):
        """Tỉ lệ khung thấy tay trái/phải/vai/mặt và số lần một bàn tay duy nhất nhảy giữa khối Left và Right
        (nhãn handedness của MediaPipe bị lật) trong chuỗi 30 khung đưa vào model."""
        stats = {k: round(v, 3) for k, v in V.sequence_stats(raw).items()}
        left = np.any(raw[:, V.LH] != 0, axis=1)
        right = np.any(raw[:, V.RH] != 0, axis=1)
        only_l, only_r = left & ~right, right & ~left
        stats["hand_swaps"] = int(np.sum((only_l[:-1] & only_r[1:]) | (only_r[:-1] & only_l[1:])))
        return stats

    def predict_frames_with_stats(self, frames):
        """Như predict_frames nhưng trả về (top5 hoặc None nếu không thấy tay, thống kê landmark)."""
        self.load()
        if not frames:
            raise ValueError("Không có khung hình hợp lệ")
        import mediapipe as mp
        with self.lock:
            hd, pd, fd = self._get_detectors()
            feats = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                feats.append(self.V.to_feature(hd.detect(img), pd.detect(img), fd.detect(img)))
            wanted = np.rint(np.linspace(0, len(feats) - 1, 30)).astype(int)
            raw = self.V.normalize_sequence(np.asarray(feats, np.float32)[wanted])[0]
            stats = self.landmark_stats(self.V, raw)
            if not np.any(raw[:, :126]):
                return None, stats
            return self._predict_raw(raw), stats

    def predict_file(self, path):
        self.load()
        with self.lock:
            hd, pd, fd = self._get_detectors()
            raw = self.V.process_video(str(path), hd, pd, fd, 30, False)
            return self._predict_raw(raw)

    def split_video(self, path, motion_threshold=0.012, idle_seconds=0.65, min_seconds=0.35, max_seconds=4.5):
        """Notebook baseline: grayscale frame difference + gap grouping, not a trained boundary detector."""
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps < 1:
            fps = 25.0
        energies, previous, count = [], None, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 72))
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                energies.append(0.0 if previous is None else float(np.mean(cv2.absdiff(gray, previous)) / 255.0))
                previous = gray
                count += 1
        finally:
            cap.release()
        if count == 0:
            raise ValueError("Video không đọc được")
        gap = max(1, round(idle_seconds * fps))
        min_len, max_len = round(min_seconds * fps), max(1, round(max_seconds * fps))
        inds = np.flatnonzero(np.asarray(energies) >= motion_threshold)
        if len(inds) == 0:
            return [], fps
        groups = []
        start = last = int(inds[0])
        for ix in inds[1:]:
            ix = int(ix)
            if ix - last > gap:
                groups.append((start, last))
                start = ix
            last = ix
        groups.append((start, last))
        segments = []
        for a, b in groups:
            a = max(0, a - round(0.2 * fps))
            b = min(count - 1, b + round(0.25 * fps))
            if b - a + 1 < min_len:
                continue
            for s in range(a, b + 1, max_len):
                e = min(b + 1, s + max_len)
                if e - s >= min_len:
                    segments.append((s, e))
        return segments, fps

    def predict_continuous(self, path, confidence=0.4):
        segments, fps = self.split_video(path)
        if not segments:
            return []
        cap = cv2.VideoCapture(str(path))
        result = []
        try:
            for a, b in segments:
                cap.set(cv2.CAP_PROP_POS_FRAMES, a)
                frames = []
                for _ in range(b - a):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                if len(frames) < 3:
                    continue
                try:
                    # Use the same 30-frame resampling and normalization as V5.
                    sampled = [frames[int(i)] for i in np.rint(np.linspace(0, len(frames)-1, 30)).astype(int)]
                    top5 = self.predict_frames(sampled)
                except ValueError as exc:
                    logger.info("Segment skipped: %s", exc)
                    continue
                if top5[0]["confidence"] >= confidence:
                    result.append({"start_sec": round(a / fps, 2), "end_sec": round(b / fps, 2),
                                   "gloss": top5[0]["gloss"], "confidence": top5[0]["confidence"], "top5": top5})
        finally:
            cap.release()
        return result
