import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from logging.handlers import RotatingFileHandler

from .inference import Predictor
from .live import LiveSegmenter

BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE / ".env")
(BASE / "logs").mkdir(exist_ok=True)
logger = logging.getLogger("visign")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(BASE / "logs" / "visign.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
DB = BASE / "visign_history.sqlite3"
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "80"))
MAX_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "120"))
CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.40"))
LIVE_MOTION_THRESHOLD = float(os.getenv("LIVE_MOTION_THRESHOLD", "0.012"))
live_clock = time.monotonic  # thời điểm nhận frame webcam; tách riêng để test thay được


def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def save_history(mode, name, segments, sentence, elapsed):
    with db() as conn:
        conn.execute("INSERT INTO history(id,created_at,mode,filename,segments,sentence,elapsed_ms) VALUES(?,?,?,?,?,?,?)",
                     (str(uuid.uuid4()), time.strftime("%Y-%m-%d %H:%M:%S"), mode, name,
                      json.dumps(segments, ensure_ascii=False), sentence, round(elapsed * 1000)))


def fallback(glosses):
    s = " ".join(str(g).strip() for g in glosses if str(g).strip())
    return s[:1].upper() + s[1:] + "." if s else ""


def translate(glosses):
    baseline = fallback(glosses)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or not glosses:
        return baseline, "fallback"
    try:
        from groq import Groq
        response = Groq(api_key=key, timeout=18).chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {"role": "system", "content": "Chuyển gloss ngôn ngữ ký hiệu Việt Nam thành câu tiếng Việt tự nhiên. Không thêm thông tin. Chỉ trả về câu dịch."},
                {"role": "user", "content": " / ".join(glosses)},
            ], temperature=0, max_completion_tokens=300)
        content = response.choices[0].message.content
        return (content.strip() if content else baseline), "groq"
    except Exception as exc:
        logger.warning("Groq translation unavailable: %s", type(exc).__name__)
        return baseline, "fallback"


@asynccontextmanager
async def lifespan(app):
    with db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS history (id TEXT PRIMARY KEY, created_at TEXT, mode TEXT, filename TEXT, segments TEXT, sentence TEXT, elapsed_ms INTEGER)")
    yield

app = FastAPI(title="ViSign AI V5", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
predictor = Predictor()


@app.exception_handler(Exception)
async def error_handler(request, exc):
    logger.exception("Unhandled error at %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Lỗi hệ thống. Kiểm tra backend/logs/visign.log"})


@app.get("/api/health")
def health():
    return {"status": "ready" if predictor.ready else "not_loaded", "model": "V5 Face Mesh", "model_error": predictor.error,
            "labels": len(predictor.labels), "groq_configured": bool(os.getenv("GROQ_API_KEY"))}


@app.get("/api/info")
def info():
    return {"config": predictor.config, "metrics": predictor.metrics, "labels": len(predictor.labels)}


@app.get("/api/labels")
def labels():
    return {"labels": predictor.labels}


class Translation(BaseModel):
    glosses: list[str] = Field(max_length=100)


@app.post("/api/translate")
def translate_api(body: Translation):
    sentence, source = translate(body.glosses)
    return {"sentence": sentence, "source": source}


@app.get("/api/history")
def history(limit: int = 30):
    limit = max(1, min(limit, 100))
    with db() as conn:
        rows = conn.execute("SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return {"items": [{**dict(r), "segments": json.loads(r["segments"])} for r in rows]}


@app.delete("/api/history")
def clear_history():
    with db() as conn:
        conn.execute("DELETE FROM history")
    return {"ok": True}


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(422, "Video không hợp lệ hoặc codec không được hỗ trợ")
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or count <= 0:
        raise HTTPException(422, "Không đọc được số khung hình hoặc FPS")
    duration = count / fps
    if duration > MAX_SECONDS:
        raise HTTPException(413, f"Video vượt quá {MAX_SECONDS:g} giây")
    return duration


@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...), mode: str = "continuous"):
    if mode not in {"continuous", "single"}:
        raise HTTPException(422, "mode phải là single hoặc continuous")
    if not (file.filename or "").lower().endswith((".mp4", ".mov", ".avi", ".webm")):
        raise HTTPException(415, "Chỉ hỗ trợ MP4, MOV, AVI hoặc WEBM")
    start = time.perf_counter()
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
            path = Path(tmp.name)
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MB * 1024 * 1024:
                    raise HTTPException(413, f"Video vượt quá {MAX_MB} MB")
                tmp.write(chunk)
        duration = video_meta(path)
        if mode == "single":
            top5 = await asyncio.to_thread(predictor.predict_file, path)
            segments = [{"start_sec": 0, "end_sec": round(duration, 2), "gloss": top5[0]["gloss"],
                         "confidence": top5[0]["confidence"], "top5": top5}] if top5[0]["confidence"] >= CONFIDENCE else []
        else:
            segments = await asyncio.to_thread(predictor.predict_continuous, path, CONFIDENCE)
        glosses = [s["gloss"] for s in segments]
        sentence, source = translate(glosses)
        elapsed = time.perf_counter() - start
        save_history(mode, file.filename, segments, sentence, elapsed)
        logger.info("video mode=%s duration=%.2f segments=%d elapsed=%.2fs", mode, duration, len(segments), elapsed)
        return {"segments": segments, "glosses": glosses, "sentence": sentence, "translation_source": source,
                "elapsed_ms": round(elapsed * 1000), "duration_sec": round(duration, 2)}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Video inference failed")
        raise HTTPException(503, f"Không thể xử lý video: {type(exc).__name__}. Kiểm tra logs.") from exc
    finally:
        await file.close()
        if path:
            path.unlink(missing_ok=True)


def decode_sampled(payloads, n=30):
    """Lấy đều n frame trên cả đoạn (giống cách lấy mẫu lúc huấn luyện) rồi mới giải mã JPEG màu."""
    wanted = np.rint(np.linspace(0, len(payloads) - 1, n)).astype(int)
    cache, frames, last = {}, [], None
    for i in wanted.tolist():
        if i not in cache:
            cache[i] = cv2.imdecode(np.frombuffer(payloads[i], dtype=np.uint8), cv2.IMREAD_COLOR)
        if cache[i] is not None:
            last = cache[i]
        frames.append(last)  # frame lỗi -> dùng frame hợp lệ gần nhất trước đó
    valid = next((f for f in frames if f is not None), None)
    if valid is None:
        raise ValueError("Không giải mã được khung hình của đoạn ký")
    return [valid if f is None else f for f in frames]


@app.websocket("/api/ws/live")
async def live(ws: WebSocket):
    """Browser streams JPEG frames continuously (~15 fps). Frames are grouped into sign segments by motion
    (LiveSegmenter, same heuristic as the uploaded-video path); each finished segment is sampled evenly to
    30 frames and classified in a worker task, so receiving and segmenting never stall during inference.
    Messages sent: {"status": "idle"|"recording"|"processing"}, {"top5", "prediction", ...} per segment,
    {"skipped": reason} when a segment has no usable hands, {"error": message} otherwise.
    Note: each segment is an isolated-gloss prediction, not a trained continuous model.
    """
    await ws.accept()
    segmenter = LiveSegmenter(motion_threshold=LIVE_MOTION_THRESHOLD)
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    send_lock = asyncio.Lock()

    async def send(obj):
        async with send_lock:
            await ws.send_json(obj)

    async def worker():
        while True:
            payloads, duration = await queue.get()
            await send({"status": "processing"})
            try:
                frames = await asyncio.to_thread(decode_sampled, payloads)
                top5, stats = await asyncio.to_thread(predictor.predict_frames_with_stats, frames)
                diagnostics = {**stats, "frames": len(payloads), "duration_sec": round(duration, 2),
                               "fps": round((len(payloads) - 1) / duration, 1) if duration > 0 else None,
                               "width": int(frames[0].shape[1]), "height": int(frames[0].shape[0])}
                logger.info("webcam segment gloss=%s conf=%s diag=%s", top5[0]["gloss"] if top5 else None,
                            top5[0]["confidence"] if top5 else None, json.dumps(diagnostics))
                if top5 is None:
                    await send({"skipped": "Không phát hiện được bàn tay trong video", "diagnostics": diagnostics})
                else:
                    label = top5[0] if top5[0]["confidence"] >= CONFIDENCE else None
                    await send({"top5": top5, "prediction": label, "diagnostics": diagnostics})
            except ValueError as exc:
                await send({"skipped": str(exc)})
            except Exception as exc:
                logger.warning("Webcam prediction error: %s", exc)
                await send({"error": f"Không thể nhận diện đoạn vừa ký: {type(exc).__name__}"})
            await send({"status": "recording" if segmenter.active else "idle"})

    task = asyncio.create_task(worker())
    try:
        await send({"status": "idle"})
        while True:
            data = await ws.receive_bytes()
            gray = None if len(data) > 2_000_000 else cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                await send({"error": "Khung hình quá lớn hoặc không đọc được"})
                continue
            now = live_clock()
            was_active = segmenter.active
            for segment in segmenter.push(now, (now, data), gray):
                if queue.full():
                    queue.get_nowait()  # inference is behind: drop the oldest pending segment, keep the newest
                queue.put_nowait(([d for _, d in segment], segment[-1][0] - segment[0][0]))
            if segmenter.active != was_active:
                await send({"status": "recording" if segmenter.active else "idle"})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Websocket failed")
    finally:
        task.cancel()
