import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import main
from app.live import LiveSegmenter

FPS = 15
STILL = np.full((240, 320), 120, np.uint8)


def moving(i):
    """Frame khác hẳn frame trước (một khối sáng di chuyển) -> năng lượng chuyển động lớn."""
    img = STILL.copy()
    x = (i * 40) % 240
    img[60:180, x:x + 80] = 250
    return img


def feed(seg, frames, start=0):
    out = []
    for k, img in enumerate(frames):
        i = start + k
        out += seg.push(i / FPS, i, img)
    return out


def test_still_camera_produces_no_segment():
    seg = LiveSegmenter()
    assert feed(seg, [STILL] * (FPS * 10)) == []
    assert not seg.active


def test_sign_is_cut_at_its_own_boundaries():
    seg = LiveSegmenter()
    frames = [STILL] * 30 + [moving(i) for i in range(30)] + [STILL] * 30   # 2 s nghỉ, 2 s ký, 2 s nghỉ
    out = feed(seg, frames)
    assert len(out) == 1
    first, last = out[0][0], out[0][-1]
    # bắt đầu ngay trước cử động (đệm <= 0,2 s), kết thúc ngay sau cử động (đệm <= 0,25 s).
    # Frame 60 (frame tĩnh đầu tiên) vẫn khác frame trước nên là frame chuyển động cuối cùng.
    assert 30 - 0.2 * FPS <= first <= 30
    assert 60 <= last <= 60 + 0.25 * FPS
    assert not seg.active


def test_two_signs_separated_by_pause_are_two_segments():
    seg = LiveSegmenter()
    frames = [STILL] * 15 + [moving(i) for i in range(20)] + [STILL] * 20 + [moving(i) for i in range(20)] + [STILL] * 20
    assert len(feed(seg, frames)) == 2


def test_short_pause_inside_a_sign_does_not_split_it():
    seg = LiveSegmenter()
    frames = [STILL] * 15 + [moving(i) for i in range(15)] + [STILL] * 5 + [moving(i) for i in range(15)] + [STILL] * 20
    assert len(feed(seg, frames)) == 1


def test_continuous_motion_is_cut_at_max_length():
    seg = LiveSegmenter(max_seconds=4.5)
    out = feed(seg, [STILL] + [moving(i) for i in range(FPS * 10)])
    assert len(out) == 2
    assert all(len(s) <= 4.5 * FPS + 2 for s in out)  # 4,5 s tính cả frame đầu và cuối
    assert seg.active


def test_blip_shorter_than_min_length_is_dropped():
    seg = LiveSegmenter(min_seconds=0.35, pre_roll=0, post_roll=0)
    assert feed(seg, [STILL] * 10 + [moving(0)] + [STILL] * 20) == []


def jpeg(img):
    return cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[1].tobytes()


def test_websocket_predicts_once_per_sign(monkeypatch):
    calls = []

    def fake_predict(frames):
        calls.append(len(frames))
        top5 = [{"gloss": "Ăn", "confidence": 0.9}] + [{"gloss": f"g{i}", "confidence": 0.025} for i in range(4)]
        return top5, {"left_rate": 0.5, "right_rate": 0.9, "pose_rate": 1.0, "face_rate": 1.0,
                      "any_hand_rate": 0.9, "hand_swaps": 1}

    monkeypatch.setattr(main.predictor, "predict_frames_with_stats", fake_predict)
    clock = iter(np.arange(0, 100, 1 / FPS))
    monkeypatch.setattr(main, "live_clock", lambda: float(next(clock)))
    frames = [STILL] * 15 + [moving(i) for i in range(30)] + [STILL] * 20
    with TestClient(main.app) as client, client.websocket_connect("/api/ws/live") as ws:
        assert ws.receive_json() == {"status": "idle"}
        for img in frames:
            ws.send_bytes(jpeg(img))
        messages = []
        while not any("top5" in m for m in messages):
            messages.append(ws.receive_json())
    assert {"status": "recording"} in messages
    result = next(m for m in messages if "top5" in m)
    assert result["prediction"]["gloss"] == "Ăn"
    assert calls == [30]   # đoạn được lấy đều đúng 30 frame như lúc huấn luyện
    diag = result["diagnostics"]
    assert diag["any_hand_rate"] == 0.9 and diag["hand_swaps"] == 1
    assert (diag["width"], diag["height"]) == (320, 240)
    assert 1.5 < diag["duration_sec"] < 3 and abs(diag["fps"] - FPS) < 0.5


def test_landmark_stats_counts_handedness_flips():
    import importlib.util
    spec = importlib.util.spec_from_file_location("vf", main.BASE / "models" / "visign_features.py")
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    raw = np.zeros((30, 204), np.float32)
    raw[0:10, V.LH] = 1      # một tay, nhãn Left
    raw[10:20, V.RH] = 1     # cùng tay đó bị gắn nhãn Right -> lật 1 lần
    raw[20:30, V.LH] = 1     # lật lại -> 2 lần
    raw[:, V.FC] = 1
    stats = main.Predictor.landmark_stats(V, raw)
    assert stats["hand_swaps"] == 2
    assert stats["any_hand_rate"] == 1.0 and stats["face_rate"] == 1.0 and stats["pose_rate"] == 0.0
    assert abs(stats["left_rate"] - 0.667) < 1e-3
