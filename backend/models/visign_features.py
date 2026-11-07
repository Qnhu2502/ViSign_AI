"""
visign_features.py - Video -> X (30, 204). Dùng CHUNG cho huấn luyện và suy luận để hai bên luôn khớp nhau.

Bố cục 204 đặc trưng mỗi frame (mỗi điểm gồm x, y, z):
    [  0: 63]  tay "Left"  (21 điểm)   theo nhãn handedness của MediaPipe
    [ 63:126]  tay "Right" (21 điểm)
    [126:144]  pose 6 điểm: vai, khuỷu, cổ tay
    [144:204]  face 20 điểm: mắt, lông mày, miệng
Điểm không phát hiện được = 0 (cả khối). Nhãn Left/Right của MediaPipe giả định ảnh gương (selfie); ta chỉ cần
dùng NHẤT QUÁN giữa huấn luyện và suy luận, không cần nó khớp với tay thật.

Chuẩn hoá: tâm = trung điểm hai vai, tỉ lệ = bề rộng vai (trung vị theo video).
"""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

T = 30
D = 204
POSE_IDS = [11, 12, 13, 14, 15, 16]
FACE_IDS = [33,133,362,263,159,145,386,374,70,63,105,107,336,334,293,300,61,291,13,14]
LH, RH, PS, FC = slice(0, 63), slice(63, 126), slice(126, 144), slice(144, 204)
BLOCKS = (LH, RH, PS, FC)
SIDE = {"Left": 0, "Right": 1}

MODEL_DIR = Path(os.environ.get("VISIGN_MODEL_DIR", "/content/mp_models"))
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
            "hand_landmarker.task")
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/"
            "pose_landmarker_lite.task")
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
_TS = [0]   # timestamp cho chế độ tracking (VIDEO)


def ensure_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in ((HAND_MODEL, HAND_URL), (POSE_MODEL, POSE_URL), (FACE_MODEL, FACE_URL)):
        if not path.exists() or path.stat().st_size == 0:
            print("Downloading:", path.name)
            urllib.request.urlretrieve(url, path)


def make_detectors(use_tracking=False):
    """use_tracking=False: chế độ IMAGE, mỗi frame độc lập, không có trạng thái giữa các video (khuyến nghị cho
    clip ngắn). use_tracking=True: chế độ VIDEO (có tracking), cần timestamp tăng dần."""
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    ensure_models()
    mode = vision.RunningMode.VIDEO if use_tracking else vision.RunningMode.IMAGE
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)), running_mode=mode, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)), running_mode=mode, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)), running_mode=mode,
        num_faces=1, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        output_face_blendshapes=False, output_facial_transformation_matrixes=False))
    return hand, pose, face


# ----------------------------------------------------------------------------
# Landmark -> vector 144 chiều (chưa chuẩn hoá)
# ----------------------------------------------------------------------------
def to_feature(hand_result, pose_result, face_result):
    f = np.zeros(D, np.float32)
    best = {}                                   # side -> (score, landmarks); mỗi bên giữ bàn tay điểm cao nhất
    hands, handed = hand_result.hand_landmarks, getattr(hand_result, "handedness", None) or []
    for k, lms in enumerate(hands[:2]):
        if len(lms) != 21 or k >= len(handed) or not handed[k]:
            continue
        cat = handed[k][0]
        side = SIDE.get(cat.category_name)
        if side is not None and (side not in best or cat.score > best[side][0]):
            best[side] = (cat.score, lms)
    for side, (_, lms) in best.items():
        f[side * 63:(side + 1) * 63] = np.array([[p.x, p.y, p.z] for p in lms], np.float32).reshape(-1)
    if pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]
        if len(lms) > max(POSE_IDS):
            f[PS] = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in POSE_IDS], np.float32).reshape(-1)
    if face_result.face_landmarks:
        lms = face_result.face_landmarks[0]
        if len(lms) > max(FACE_IDS):
            f[FC] = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in FACE_IDS], np.float32).reshape(-1)
    return f


# ----------------------------------------------------------------------------
# Chuẩn hoá theo cơ thể
# ----------------------------------------------------------------------------
def body_reference(arr):
    """(tâm (3,), tỉ lệ) từ hai vai (trung vị theo video); None nếu không có frame nào thấy pose."""
    ok = np.any(arr[:, PS] != 0, axis=1)
    if not ok.any():
        return None
    pose = arr[ok, PS].reshape(-1, 6, 3)
    ls, rs = pose[:, 0], pose[:, 1]
    scale = float(np.median(np.linalg.norm((ls - rs)[:, :2], axis=1)))
    if scale < 0.02:
        return None
    return np.median((ls + rs) / 2, axis=0).astype(np.float32), scale


def normalize_sequence(arr):
    """(T,204) toạ độ ảnh -> toạ độ theo thân người. Điểm thiếu giữ nguyên 0.
    Pose: (v - tâm) / tỉ lệ. Tay: x,y theo (v - tâm) / tỉ lệ; z chỉ chia tỉ lệ (z của tay tính so với cổ tay).
    Trả về (mảng đã chuẩn hoá, đã_dùng_vai: bool). Không thấy vai -> dùng tâm (0.5, 0.5), tỉ lệ 0.25."""
    ref = body_reference(arr)
    found = ref is not None
    center, scale = ref if found else (np.array([0.5, 0.5, 0.0], np.float32), 0.25)
    out = np.zeros_like(arr, dtype=np.float32)
    n = len(arr)
    for sl in (LH, RH):
        present = np.any(arr[:, sl] != 0, axis=1)
        h = arr[:, sl].reshape(n, 21, 3).copy()
        h[..., :2] = (h[..., :2] - center[:2]) / scale
        h[..., 2] = h[..., 2] / scale
        h[~present] = 0
        out[:, sl] = h.reshape(n, 63)
    present = np.any(arr[:, PS] != 0, axis=1)
    p = (arr[:, PS].reshape(n, 6, 3) - center) / scale
    p[~present] = 0
    out[:, PS] = p.reshape(n, 18)
    face_present = np.any(arr[:, FC] != 0, axis=1)
    face_pts = arr[:, FC].reshape(n, len(FACE_IDS), 3).copy()
    face_pts[..., :2] = (face_pts[..., :2] - center[:2]) / scale
    face_pts[..., 2] = face_pts[..., 2] / scale
    face_pts[~face_present] = 0
    out[:, FC] = face_pts.reshape(n, 60)
    return np.clip(out, -8.0, 8.0), found


# ----------------------------------------------------------------------------
# Video -> (30, 144)
# ----------------------------------------------------------------------------
def read_selected(path, n_samples=T):
    """Đọc đúng n_samples frame trải đều trên video (đếm frame thật, không tin CAP_PROP_FRAME_COUNT).
    Trả về (dict chỉ_số -> frame BGR, danh sách chỉ_số_cần_lấy có thể lặp khi video ngắn)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    if n <= 0:
        raise ValueError(f"Empty video: {path}")
    wanted = np.linspace(0, n - 1, n_samples).round().astype(int)
    need = set(wanted.tolist())
    cap = cv2.VideoCapture(str(path))
    frames, idx = {}, 0
    while idx < n and len(frames) < len(need):
        if not cap.grab():
            break
        if idx in need:
            ok, f = cap.retrieve()
            if ok:
                frames[idx] = f
        idx += 1
    cap.release()
    if not frames:
        raise ValueError(f"No decodable frames: {path}")
    last = None
    fixed = []
    for w in wanted:                         # frame không đọc được -> dùng frame hợp lệ gần nhất trước đó
        if int(w) in frames:
            last = int(w)
        elif last is None:
            last = min(frames)
        fixed.append(last)
    return frames, fixed


def process_video(path, hand, pose, face, T_=T, use_tracking=False, return_ref_flag=False):
    import mediapipe as mp
    frames, wanted = read_selected(path, T_)
    feats = {}
    for idx in sorted(set(wanted)):
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if use_tracking:
            _TS[0] += 33
            hr, pr, fr = hand.detect_for_video(img, _TS[0]), pose.detect_for_video(img, _TS[0]), face.detect_for_video(img, _TS[0])
        else:
            hr, pr, fr = hand.detect(img), pose.detect(img), face.detect(img)
        feats[idx] = to_feature(hr, pr, fr)
    arr = np.stack([feats[i] for i in wanted])
    arr, found = normalize_sequence(arr)
    if arr.shape != (T_, D) or not np.isfinite(arr).all():
        raise ValueError(f"Invalid features {arr.shape} for {path}")
    return (arr, found) if return_ref_flag else arr


def sequence_stats(arr):
    """Tỉ lệ frame phát hiện được tay trái/phải/pose của một chuỗi (T,144) đã chuẩn hoá."""
    pres = [np.any(arr[:, sl] != 0, axis=1).mean() for sl in BLOCKS]
    return dict(left_rate=float(pres[0]), right_rate=float(pres[1]), pose_rate=float(pres[2]), face_rate=float(pres[3]),
                any_hand_rate=float(np.any([np.any(arr[:, s] != 0, axis=1) for s in (LH, RH)], axis=0).mean()))


# ----------------------------------------------------------------------------
# Scaler chỉ dùng giá trị CÓ THẬT (điểm thiếu = 0 không làm lệch mean/std)
# ----------------------------------------------------------------------------
def present_mask(X):
    """Mặt nạ bool cùng shape X: True nếu cả khối (tay trái / tay phải / pose) có dữ liệu ở frame đó."""
    M = np.zeros(np.shape(X), bool)
    for sl in BLOCKS:
        M[..., sl] = np.any(np.asarray(X)[..., sl] != 0, axis=-1, keepdims=True)
    return M


def fit_masked_scaler(X_train):
    """Trả về sklearn StandardScaler (lưu được bằng joblib như bình thường) với mean/std tính trên giá trị có thật."""
    from sklearn.preprocessing import StandardScaler
    Xf = np.asarray(X_train, np.float64).reshape(-1, D)
    M = present_mask(X_train).reshape(-1, D)
    cnt = M.sum(0)
    mean = np.where(cnt > 0, (Xf * M).sum(0) / np.maximum(cnt, 1), 0.0)
    var = np.where(cnt > 0, (((Xf - mean) ** 2) * M).sum(0) / np.maximum(cnt, 1), 1.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    sc = StandardScaler()
    sc.mean_, sc.scale_, sc.var_ = mean, std, std ** 2
    sc.n_features_in_, sc.n_samples_seen_ = D, int(cnt.max())
    return sc


def apply_scaler(X, scaler):
    """Chuẩn hoá (..., T, 204); điểm thiếu giữ nguyên 0 sau chuẩn hoá."""
    X = np.asarray(X, np.float32)
    M = present_mask(X)
    out = ((X - scaler.mean_) / scaler.scale_).astype(np.float32)
    out[~M] = 0.0
    return out


def scale_array(X, scaler=None):
    """Tương thích với tên hàm cũ trong notebook. Cần truyền scaler (hoặc gán scale_array.scaler)."""
    sc = scaler if scaler is not None else getattr(scale_array, "scaler", None)
    if sc is None:
        raise RuntimeError("Chưa có scaler: gọi scale_array(X, scaler) hoặc đặt scale_array.scaler = scaler")
    return apply_scaler(X, sc)
