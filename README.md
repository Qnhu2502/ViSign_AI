# VSL Live - nhận diện ngôn ngữ ký hiệu Việt Nam qua webcam và video

Web app nhận diện ký hiệu Việt Nam, backend FastAPI, frontend JavaScript thuần (không cần build). MediaPipe trích tay, thân trên và mặt, BiLSTM + Attention (V5, 100 gloss) nhận diện từng ký hiệu, rồi Groq ghép chuỗi gloss thành câu tiếng Việt. Người dùng bật webcam hoặc tải video MP4 lên. Em đo độ chính xác trên **ký hiệu đơn**; phần ghép nhiều ký hiệu thành câu là baseline, em chưa đo (xem mục "Giới hạn").


## 1. Backend
```bash
python -m venv .venv && .venv\Scripts\activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt                      # đủ tensorflow 2.21.0, mediapipe 0.10.32, scikit-learn 1.6.1
python scripts/download_models.py                    # tải 3 file .task của MediaPipe vào models/mediapipe/
copy .env.example .env                               # Linux/macOS: cp .env.example .env
python run.py
```
- Phiên bản thư viện ghim theo `models/versions.json` của gói V5. `best_model.keras` lưu bằng Keras 3.13, còn `scaler.pkl` cần đúng scikit-learn 1.6.1 để đọc không cảnh báo.
- Lần chạy đầu cần Internet để tải 3 file `.task` (hand, pose lite, face). Nếu mạng chặn, anh tải thủ công rồi đặt vào `models/mediapipe/`.
- Model nạp ở nền sau khi khởi động. Lần đầu có thể mất vài chục giây, trang **Trạng thái hệ thống** cho biết khi nào sẵn sàng, nếu lỗi sẽ nêu nguyên nhân kèm nút "Tải lại mô hình".
- Swagger ở http://127.0.0.1:8000/api/docs, còn `GET /api/status` cho biết model, MediaPipe, Groq và SQLite đang ra sao.

## 2. Frontend
 FastAPI phục vụ luôn thư mục `frontend/` nên chỉ cần mở **http://127.0.0.1:8000**.
- Camera chỉ mở được trên `localhost`/`127.0.0.1` hoặc HTTPS. Khi demo anh dùng `http://127.0.0.1:8000`, đừng dùng địa chỉ IP trong mạng nội bộ bằng HTTP.
- Webcam dùng WebSocket nên cần `uvicorn[standard]` (đã nằm trong `requirements.txt`). Báo "mất kết nối" thường là do thiếu gói này.
- Giao diện đã kiểm tra ở độ rộng 1280 và 390 px (không tràn ngang).

## 3. File môi trường
- `.env` (mẫu ở `.env.example`, **không commit**):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GROQ_API_KEY` | trống | khóa Groq, lấy tại console.groq.com. Trống thì dịch bằng cách ghép gloss theo quy tắc |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | mô hình dịch |
| `STUDENT_NAME`, `ADVISOR_NAME` | trống | hiện ở chân trang chủ (trống thì hiện `[TÊN SINH VIÊN]` và `[TÊN GIẢNG VIÊN]`) |
| `VISIGN_HOST`, `VISIGN_PORT` | `127.0.0.1`, `8000` | địa chỉ chạy |
| `VISIGN_LOG_LEVEL` | `INFO` | mức log |
| `VISIGN_MAX_UPLOAD_MB` | `100` | dung lượng video tối đa |
| `VISIGN_MAX_VIDEO_SECONDS` | `180` | thời lượng video tối đa |
| `VISIGN_MODEL_DIR` | `models/mediapipe` | nơi chứa 3 file `.task` |
| `VISIGN_ASSET_DIR` | `models` | nơi chứa `best_model.keras`, `scaler.pkl`, `label_encoder.pkl`, `config.json` |

- Cài đặt chỉnh được khi đang chạy (trang **Cài đặt**) lưu trong SQLite. Form được dựng từ `backend/app/settings_spec.py`, muốn thêm một cài đặt chỉ cần thêm một dòng ở đó.
- Số liệu trên trang chủ (độ chính xác, số ký hiệu) đọc từ `models/config.json` chứ không gõ cứng.

## Cài đặt chính (mặc định)
| Nhóm | Cài đặt | Mặc định |
|---|---|---|
| Nhận diện | Ngưỡng độ tin cậy (dự đoán thấp hơn không vào chuỗi gloss) | 0,5 |
| | Tự động tách ký hiệu (webcam) | bật |
| | Ngưỡng chuyển động / nghỉ kết thúc ký hiệu | 0,018 / 0,8 giây |
| | Độ dài một ký hiệu (tối thiểu / tối đa) | 0,5 / 4,0 giây |
| Camera | Độ phân giải, số khung gửi mỗi giây (tối đa) | 640x480, 15 |
| | Lật ngang ảnh trước khi vào mô hình | tắt (giống `webcam_multigloss.py` của V5) |
| Video | Chế độ | `multi` (tách nhiều ký hiệu) hoặc `single` (cả video là một ký hiệu) |
| | Ngưỡng chuyển động / khoảng nghỉ giữa hai ký hiệu | 0,012 / 0,65 giây |
| Dịch | Bật Groq, temperature, token tối đa | bật, 0, 512 |
| Lịch sử | Tự lưu sau khi nhận diện video | bật |

Em giữ "Lật ngang ảnh" tắt vì lúc huấn luyện và chạy thử notebook, ảnh đưa vào MediaPipe là ảnh gốc từ `cap.read()`, không lật. Nếu nhận diện kém bất thường và trang Webcam báo nhãn tay đổi trái/phải liên tục, anh thử bật tùy chọn này.

## Các trang
| Trang | Chức năng |
|---|---|
| Trang chủ (`#/`) | Giới thiệu, số liệu từ `config.json` |
| Tổng quan (`#/dashboard`) | Trạng thái model, độ chính xác test, thống kê sử dụng |
| Webcam (`#/live`) | Webcam, vẽ khung xương, chuỗi gloss, Top-3 và độ tin cậy, mục chẩn đoán (khung/giây, tỉ lệ có tay) |
| Video (`#/video`) | Tải video, tiến trình, dòng thời gian các đoạn, chi tiết từng đoạn |
| Dịch câu (`#/translate`) | Dựng chuỗi gloss, dịch bằng Groq, tự rơi về ghép theo quy tắc nếu Groq lỗi (có ghi rõ) |
| Từ điển (`#/dictionary`) | 100 nhãn, tìm không dấu, chọn nhiều nhãn để chuyển sang Dịch câu |
| Lịch sử (`#/history`) | Tìm kiếm, lọc nguồn và ngày, xem chi tiết, xóa, xuất CSV/JSON |
| Cài đặt (`#/settings`) | Các cài đặt ở bảng trên, kiểm tra kết nối Groq |
| Trạng thái (`#/status`) | Model, MediaPipe, Groq, SQLite, hiệu năng, nhật ký lỗi |

## Luồng hoạt động
1. **Webcam.** Trình duyệt gửi từng khung JPEG (không lật) qua `WS /ws/live`, chỉ gửi khung kế tiếp sau khi nhận phản hồi khung trước nên không dồn hàng đợi. Server trích landmark và trả về để trình duyệt vẽ khung xương.
2. **Tách ký hiệu.** Ở chế độ tự động, bộ tách theo chuyển động cổ tay và khuỷu cắt luồng khung thành từng đoạn (nghỉ đủ lâu thì chốt đoạn). Tắt chế độ này thì anh tự bấm Bắt đầu/Dừng cho từng ký hiệu.
3. **Nhận diện.** Mỗi đoạn được lấy mẫu đều thành đúng 30 khung, chuẩn hóa theo vai (tâm là trung điểm hai vai, tỉ lệ là bề rộng vai), qua `scaler.pkl`, rồi vào model. Trả Top-3 và độ tin cậy. Dự đoán dưới ngưỡng bị bỏ.
4. **Video.** `POST /api/recognize/video` trả về mã job ngay, ở nền server tách đoạn theo chênh lệch ảnh xám rồi chạy từng đoạn như bước 3. Trang Video hỏi `GET /api/jobs/{id}` để hiện tiến trình.
5. **Dịch.** Chuỗi gloss gửi `POST /api/translate`. Groq (`temperature=0`, prompt yêu cầu giữ nguyên ý, không thêm thông tin) trả câu. Chưa có khóa, tắt dịch hoặc Groq lỗi thì dùng câu ghép theo quy tắc và trang hiện cảnh báo.
6. **Lưu.** Kết quả video (nếu bật tự lưu) và các bản dịch được lưu vào SQLite, xem ở trang Lịch sử.

Đặc trưng lúc chạy web dùng nguyên `backend/app/ml/visign_features.py` của gói V5 (cùng file với notebook huấn luyện), nên không có lệch giữa train và suy luận.

## API chính
- `GET /api/status`, `GET /api/status/logs` (và `DELETE`), `POST /api/status/check-groq`, `POST /api/status/reload-model`
- `GET /api/dashboard`, `GET /api/dictionary`
- `GET|PUT /api/settings`, `POST /api/settings/reset`
- `POST /api/recognize/video` (multipart), `GET /api/jobs/{id}`
- `POST /api/translate` với `{"glosses": [...], "source": "...", "save": true}`
- `GET|POST|DELETE /api/history`, `GET|DELETE /api/history/{id}`, `GET /api/history/export?format=csv|json`. Xóa toàn bộ phải thêm `?confirm=true`.
- `WS /ws/live`

Mọi lỗi trả về cùng một dạng `{"error": {"code", "message", "request_id"}}`. Log nằm ở `data/logs/app.log` (quay vòng), cảnh báo và lỗi cũng hiện ở trang Trạng thái.

## Cấu trúc
```
backend/app/   main.py, config.py, settings_spec.py, db.py (SQLite), errors.py, logging_setup.py, schemas.py
               ml/visign_features.py        file đặc trưng gốc của V5
               services/                    model_service, features, recognizer, segmenter, live, video_service, jobs, translation
               routers/                     system, data, recognition, live (WebSocket)
backend/tests/ test_core.py, test_api.py, test_frontend_static.py, fakes.py
frontend/      index.html, css/app.css, js/{main,api,ui,skeleton}.js, js/pages/*.js
models/        best_model.keras, scaler.pkl, label_encoder.pkl, config.json, versions.json, mediapipe/
data/          visign.db, logs/ (tự tạo)
scripts/       download_models.py, reference_webcam_multigloss.py
```

## Kiểm thử
```bash
pip install -r requirements-dev.txt
pytest backend/tests
```
Hiện có 32 test đạt, 1 bỏ qua. Các test dùng **bản giả của MediaPipe và TensorFlow** (`tests/fakes.py`) nên chỉ kiểm tra đường ống: tách đoạn, phiên webcam, job video, dịch Groq (giả lập mạng), cài đặt, toàn bộ endpoint, nạp scaler và label thật, kiểm tra tĩnh giao diện. Chúng **không đo độ chính xác** của model. Muốn đánh giá thật, anh dùng video thật có nhãn rồi so kết quả.

## Giới hạn cần nói rõ khi báo cáo
- **84,21% (Top-5 95,45%)** là độ chính xác trên 418 clip *ký hiệu đơn* của tập test, khoảng tin cậy 95% là 80,4% đến 87,4%. Không phải độ chính xác dịch câu.
- Nhận diện nhiều ký hiệu (webcam liên tục, video dài) dùng tách đoạn theo chuyển động, là **baseline chưa đo**. Kết quả có thể kém hơn nhiều tùy ánh sáng, nền và tốc độ ký. Model luôn trả về một trong 100 nhãn, kể cả khi người dùng không ký gì; chỉ có ngưỡng độ tin cậy để lọc bớt.
- Webcam thường gửi ít hơn 30 khung/giây (lần thử của em ra khoảng 15 khung/giây ở 640x480), trong khi lúc huấn luyện mỗi clip được lấy mẫu 30 khung. Số khung/giây thực tế hiện ở mục chẩn đoán của trang Webcam.
- Câu Groq là diễn đạt lại chuỗi gloss, có thể thêm hoặc bớt từ.
- Nhãn tay Left/Right theo MediaPipe giả định ảnh gương, và MediaPipe không thấy tay ở khoảng 1/4 số khung của bộ dữ liệu.
- Từ điển chỉ có nhãn, chưa có video minh họa.
- Em chưa đo độ trễ ở trạng thái đã nóng máy. Lần thử video 3,27 giây của em mất 6,5 giây, gồm cả lần nạp đầu tiên.

## Xử lý sự cố
| Triệu chứng | Cách xử lý |
|---|---|
| "Chưa cài TensorFlow" | `pip install -r requirements.txt`, rồi bấm "Tải lại mô hình" ở trang Trạng thái |
| "Không khởi tạo được MediaPipe" | Chạy `python scripts/download_models.py` hoặc tải 3 file `.task` thủ công vào `models/mediapipe/` |
| Không bật được camera | Dùng `http://127.0.0.1:8000`, cho phép camera trong trình duyệt, đóng ứng dụng khác đang dùng camera |
| Webcam báo mất kết nối | Cài `uvicorn[standard]` (có `websockets`) |
| Dịch luôn là "ghép theo quy tắc" | Kiểm tra `GROQ_API_KEY`, bật "Dịch bằng Groq" trong Cài đặt, bấm "Kiểm tra kết nối" |
| Video không có đoạn nào | Giảm "Ngưỡng chuyển động (video)" hoặc dùng chế độ "Một ký hiệu" |
| Cảnh báo đọc `scaler.pkl` | Cài đúng `scikit-learn==1.6.1`. File `.pkl` chỉ tải từ bộ sản phẩm tin cậy |