# ViSign AI — VSL Live (V5 Face Mesh)

Web application dựa trên thiết kế tham khảo được cung cấp: nền trắng/xám nhạt, xanh đen, điểm nhấn vàng và cam.

## Chức năng

- Trang chủ theo thiết kế; hiển thị **số liệu thực từ config.json** (Accuracy test 84.21%, Top-5 95.45%, 100 gloss).
- Nhận diện webcam qua trình duyệt → gửi JPEG qua WebSocket → 30 frame/lượt → MediaPipe + BiLSTM Attention V5.
- Upload MP4/MOV/AVI/WEBM; nhận diện gloss đơn hoặc chuỗi gloss theo thuật toán phân đoạn chuyển động trong notebook.
- Dịch Groq `openai/gpt-oss-20b`; fallback ghép gloss khi chưa có khóa hoặc hết quota.
- Từ điển 100 nhãn lấy từ `label_encoder.pkl`; lịch sử SQLite; xuất CSV; logging quay vòng; kiểm tra trạng thái.

## Cài đặt trên Windows 11

Cần Python **3.13**, Node.js 20+ và kết nối Internet ở lần đầu để tải các mô hình `.task` MediaPipe (từ Google). Nên dùng đúng phiên bản TensorFlow/MediaPipe từ `backend/models/versions.json` nếu có thể.

### Terminal 1: backend

```powershell
cd ViSign_AI\backend
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Nếu PowerShell chặn kích hoạt môi trường ảo, có thể gọi trực tiếp `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` và `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000`.

Lần nhận diện đầu tiên có thể chậm vì TensorFlow nạp checkpoint và MediaPipe tải các model Hand/Pose/Face. **Tải model `.task` chỉ từ nguồn tin cậy**.

### Terminal 2: frontend

```powershell
cd ViSign_AI\frontend
npm install
npm run dev
```

Mở **http://127.0.0.1:5173**. API docs: http://127.0.0.1:8000/docs

### Groq

Mở `backend/.env`, điền `GROQ_API_KEY=...` (không chia sẻ khóa và không commit `.env`). Nếu không có khóa, web vẫn hiển thị gloss và ghép câu dự phòng.

## Lưu ý quan trọng

1. **84.21% là độ chính xác trên tập test ký hiệu đơn**, không phải độ chính xác dịch câu liên tục.
2. `predict_continuous` sử dụng thuật toán **phân đoạn dựa trên sai khác ảnh xám** được chuyển từ notebook. Nó chưa được huấn luyện end-to-end để tìm ranh giới gloss.
3. Webcam xử lý các **cụm 30 frame**. Kết quả có thể lặp gloss khi giữ ký hiệu; chưa có thuật toán loại trùng ổn định.
4. MediaPipe thực sự trích xuất tay, mặt, vai, khuỷu và cổ tay trên backend. Bản này **chưa vẽ các landmark động lên video**, chỉ có minh họa trên trang chủ và trạng thái theo dõi; muốn vẽ đúng điểm thực cần trả tọa độ qua WebSocket.
5. Cần kiểm thử mô hình thực trên Windows sau khi cài các dependency. Việc chạy lệnh build frontend và unit test không thay thế kiểm thử webcam thực tế.
6. File `.pkl` sử dụng joblib/pickle: chỉ tải từ bộ sản phẩm tin cậy.

## API

- `GET /api/health`, `GET /api/info`, `GET /api/labels`
- `POST /api/predict/video?mode=single|continuous` (multipart field `file`)
- `POST /api/translate` JSON `{"glosses":["Tôi","Muốn","Uống","Nước"]}`
- `GET /api/history`, `DELETE /api/history`
- `WS /api/ws/live` gửi các frame JPEG dạng binary; mỗi 30 frame nhận JSON top5.

## Cấu trúc

- `backend/models`: checkpoint và tiền xử lý V5 gốc.
- `backend/app/inference.py`: nạp model, trích đặc trưng, nhận diện và phân đoạn.
- `backend/app/main.py`: API, Groq, SQLite, logging, lỗi.
- `frontend/src`: React + TypeScript + CSS theo hai ảnh mẫu.
- `reference_home.png`, `reference_recognition.png`: giao diện tham khảo.

## Ghi chú về webcam

Trình duyệt chỉ cho phép camera trên HTTPS hoặc localhost. Trong quá trình demo, dùng `http://127.0.0.1:5173`. Nếu nhận diện bị chậm, giảm độ phân giải webcam hoặc tốc độ gửi frame. Mặc định gửi 1 frame/160ms, nên một lượt 30 frame mất tối thiểu khoảng 4.8 giây cộng thời gian suy luận. Đây là prototype theo lô, **chưa phải dịch thời gian thực độ trễ thấp**.
