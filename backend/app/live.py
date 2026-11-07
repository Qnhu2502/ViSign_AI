"""Cắt đoạn ký hiệu trực tiếp từ luồng webcam.

Dữ liệu huấn luyện là các clip ngắn đã cắt sát ký hiệu, rồi lấy đều 30 frame trên toàn clip. Vì vậy luồng webcam
không được chia thành các khối 30 frame cố định (ký hiệu sẽ bị cắt đôi hoặc lẫn thời gian đứng yên); thay vào đó
mỗi đoạn được xác định theo chuyển động, cùng heuristic với `Predictor.split_video` dùng cho video tải lên:
khác biệt ảnh xám 96x72 >= ngưỡng là có chuyển động, nghỉ quá `idle_seconds` là kết thúc đoạn,
thêm 0,2 s trước và 0,25 s sau, bỏ đoạn ngắn hơn `min_seconds`, cắt đoạn dài hơn `max_seconds`.
Khác với video tải lên, các mốc ở đây tính theo giây (thời điểm nhận frame) vì FPS webcam không cố định.
"""
from collections import deque

import cv2
import numpy as np


def motion_gray(gray):
    """Ảnh xám -> ảnh nhỏ đã làm mờ dùng để đo chuyển động (giống split_video)."""
    return cv2.GaussianBlur(cv2.resize(gray, (96, 72)), (5, 5), 0)


class LiveSegmenter:
    def __init__(self, motion_threshold=0.012, idle_seconds=0.65, min_seconds=0.35, max_seconds=4.5,
                 pre_roll=0.2, post_roll=0.25):
        self.motion_threshold = motion_threshold
        self.idle_seconds = idle_seconds
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.pre_roll = pre_roll
        self.post_roll = post_roll
        self.previous = None
        self.active = False
        self.history = deque()      # (t, payload) gần nhất khi đang nghỉ, dùng làm phần đệm trước đoạn
        self.frames = []            # (t, payload) của đoạn đang ghi
        self.last_motion = None

    def push(self, t, payload, gray):
        """Thêm một frame (thời điểm t giây, payload bất kỳ, ảnh xám). Trả về danh sách đoạn vừa hoàn tất,
        mỗi đoạn là list payload theo thứ tự thời gian."""
        small = motion_gray(gray)
        energy = 0.0 if self.previous is None else float(np.mean(cv2.absdiff(small, self.previous)) / 255.0)
        self.previous = small
        moving = energy >= self.motion_threshold
        done = []

        if not self.active:
            self.history.append((t, payload))
            while self.history and t - self.history[0][0] > self.pre_roll:
                self.history.popleft()
            if moving:
                self.active = True
                self.frames = list(self.history)
                self.history.clear()
                self.last_motion = t
            return done

        self.frames.append((t, payload))
        if moving:
            self.last_motion = t
        if t - self.last_motion > self.idle_seconds:
            end = self.last_motion + self.post_roll
            self._emit([f for f in self.frames if f[0] <= end], done)
            # Các frame nghỉ cuối đoạn trở thành phần đệm cho đoạn sau.
            self.history = deque(f for f in self.frames if t - f[0] <= self.pre_roll)
            self.frames = []
            self.active = False
        elif t - self.frames[0][0] >= self.max_seconds:
            self._emit(self.frames, done)
            self.frames = []
        return done

    def _emit(self, frames, done):
        if len(frames) >= 3 and frames[-1][0] - frames[0][0] >= self.min_seconds:
            done.append([p for _, p in frames])
