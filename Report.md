# Nhận diện ngôn ngữ ký hiệu Việt Nam động bằng MediaPipe và BiLSTM–Attention: huấn luyện, đánh giá độ tin cậy và triển khai web


## Tóm tắt

Em xây dựng hệ thống nhận diện 100 ký hiệu của ngôn ngữ ký hiệu Việt Nam từ video ngắn. Mỗi clip được lấy 30 khung hình trải đều; MediaPipe trích tọa độ hai bàn tay, sáu điểm thân trên và 20 điểm khuôn mặt (204 đặc trưng mỗi khung); chuỗi 30×204 được chuẩn hóa theo bề rộng vai rồi đưa vào mạng BiLSTM hai tầng kèm attention theo thời gian (543.077 tham số). Sau khi loại 1.101 bản ghi trùng lặp hoặc mâu thuẫn nhãn, em chia dữ liệu 70/15/15 theo nhóm video gốc và huấn luyện 83 epoch (dừng sớm, epoch tốt nhất là 63).

Trên 418 clip test, độ chính xác Top-1 khoảng **84%** (351/418 = 83,97% theo file kết quả chi tiết, khoảng tin cậy Wilson 95% là 80,1–87,2%) và Top-5 khoảng 95%. Em báo cáo thêm ba phân tích mà bản trước chưa có: (i) **độ tin cậy do mô hình trả ra khá đáng tin** (ECE = 0,044; ở ngưỡng 0,9 độ đúng đạt 96,9% trên 69% số clip), (ii) lỗi tập trung ở các clip mà MediaPipe thấy tay ít (độ đúng 65,8% khi ≤50% số khung thấy tay, so với 89,9% khi >75%), và (iii) các lớp có ít dữ liệu yếu rõ rệt (recall gộp 62,8% ở nhóm lớp chỉ có 1–2 clip test so với 87,6–89,7% ở nhóm 5–8 clip). Các kết quả trên chỉ áp dụng cho **ký hiệu đơn**. Phần ghép nhiều ký hiệu và dịch câu bằng Groq mới là đường ống chạy thông, chưa được đo độ chính xác.

## 1. Đặt vấn đề

Ngôn ngữ ký hiệu là kênh giao tiếp chính của cộng đồng người khiếm thính, nhưng công cụ nhận diện ký hiệu tiếng Việt còn ít. Bài toán của em có hai tầng. Tầng một là **nhận diện ký hiệu đơn**: cho một clip ngắn chứa đúng một ký hiệu, dự đoán nhãn trong 100 lớp. Tầng hai là **ghép chuỗi**: cắt luồng video liên tục thành các đoạn, nhận diện từng đoạn rồi nhờ mô hình ngôn ngữ lớn (LLM) diễn đạt chuỗi nhãn thành câu. Em huấn luyện và đánh giá nghiêm túc tầng một, còn tầng hai được triển khai trên web dưới dạng baseline.

Em chọn đặc trưng khung xương (MediaPipe) thay vì học trực tiếp từ điểm ảnh vì hai lý do: vector 204 chiều nhỏ gọn nên mô hình nhẹ (2,07 MB) và chạy được trên máy cá nhân, và tọa độ khung xương ít phụ thuộc vào nền, ánh sáng, màu da hơn ảnh thô. Đổi lại, chất lượng toàn hệ thống bị chặn trên bởi chất lượng phát hiện của MediaPipe, điều mà mục 4.3 sẽ định lượng.

## 2. Phương pháp

### 2.1. Dữ liệu và làm sạch

Em dùng bộ video `namwu/ViSignLanguage-Video` (Hugging Face). Bộ có ba thư mục: `train` (3.875 video có nhãn theo tên thư mục), `public_test` (1.630) và `private_test` (2.859). Hai thư mục sau không có nhãn nên em không dùng. Như vậy dữ liệu có nhãn gồm 3.875 video, 100 lớp, 2.806 video gốc (nhóm theo tên file); mỗi lớp có từ 6 đến 74 video (trung vị 45,5).

**Trùng lặp và mâu thuẫn nhãn.** Em băm SHA-256 từng file. Có 550 nhóm video giống hệt nhau từng byte, trong đó 2 nhóm mang nhãn mâu thuẫn (cùng một file nằm ở hai thư mục lớp khác nhau). Em loại cả hai nhóm mâu thuẫn và giữ một bản cho mỗi nhóm trùng còn lại. Kết quả là **2.774 video** (bỏ 1.101 bản ghi, 28,4%), vẫn đủ 100 lớp.

**Chia tập.** Em chia 70/15/15 theo nhóm video gốc trong từng lớp (seed 42), trước khi trích đặc trưng và trước khi fit scaler để không rò rỉ thông tin từ val/test sang train. Các assert trong notebook kiểm tra: không trùng byte giữa các tập, không trùng nhóm trong cùng lớp, mỗi tập đủ 100 lớp.

| Tập | Số video | Tỉ lệ | Vai trò |
|---|---:|---:|---|
| Train | 1.938 (còn 1.913 sau khi bỏ 25 clip không thấy tay) | 69,86% | học tham số |
| Validation | 418 | 15,07% | chọn checkpoint, early stopping, giảm learning rate |
| Test | 418 | 15,07% | đánh giá cuối |

**Kiểm tra trùng ở mức đặc trưng.** Sau khi trích, em băm cả ma trận đặc trưng để tìm trùng mà băm byte của file video không thấy. Có 2 nhóm đặc trưng trùng, không nhóm nào nằm vắt qua hai tập. Nhóm thứ nhất là hai clip train (`Biết/214576` và `Khai báo/885339`) có đặc trưng **toàn bằng 0**: MediaPipe không phát hiện được gì (kể cả pose và mặt), nên hai vector giống nhau dù khác nhãn. Đây chính là "nhóm trùng khác nhãn" mà cảnh báo trong notebook nhắc tới; nó không phải lỗi gán nhãn, và hai clip này thuộc nhóm bị loại khỏi train vì không thấy tay. Nhóm thứ hai là hai clip train cùng lớp `Xúc động` có landmark giống hệt nhau nhưng file khác byte, tức là bản mã hóa lại của cùng một cảnh. Nhóm này cho thấy bộ dữ liệu còn chứa các bản sao không bắt được bằng băm file; trong trường hợp này cả hai nằm trong train nên không gây rò rỉ, nhưng em không loại trừ được các cặp tương tự (không giống hệt ở mức landmark) nằm ở hai tập khác nhau.

**Chất lượng phát hiện landmark.** Tỉ lệ khung MediaPipe phát hiện được, trung bình theo tập:

| Tập | Tay trái | Tay phải | ≥1 tay | Pose | Mặt | Clip không thấy tay |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0,571 | 0,671 | 0,766 | 0,915 | 0,905 | 26 / 1.938 |
| Val | 0,541 | 0,646 | 0,740 | 0,920 | 0,912 | 10 / 418 |
| Test | 0,579 | 0,687 | 0,774 | 0,925 | 0,916 | 6 / 418 |

Tức là trung bình khoảng 23% số khung không có tay nào được phát hiện, và 42 trên 2.774 clip hoàn toàn không thấy tay. Em loại 25 trong 26 clip như vậy khỏi **train** (giữ lại một clip để mỗi lớp còn ít nhất 3 mẫu), nhưng giữ nguyên ở val và test để số đo phản ánh điều kiện thực tế.

### 2.2. Đặc trưng

Mỗi clip được đọc đúng 30 khung trải đều theo số khung thực tế (đếm khung bằng `grab()` thay vì tin vào `CAP_PROP_FRAME_COUNT`). MediaPipe chạy ở chế độ IMAGE, mỗi khung độc lập, không tracking. Vector mỗi khung có 204 chiều:

| Khối | Cột | Nội dung |
|---|---|---|
| Tay "Left" | 0–62 | 21 điểm × (x, y, z) |
| Tay "Right" | 63–125 | 21 điểm × (x, y, z) |
| Pose | 126–143 | 6 điểm: hai vai, hai khuỷu, hai cổ tay |
| Mặt | 144–203 | 20 điểm quanh mắt, lông mày, miệng |

Nhãn Left/Right do MediaPipe gán và giả định ảnh gương; em chỉ cần dùng nhất quán giữa huấn luyện và suy luận. Điểm không phát hiện được đặt bằng 0 (cả khối).

**Chuẩn hóa theo thân người.** Tâm là trung điểm hai vai, thước đo là bề rộng vai (trung vị theo video), nên đặc trưng ít phụ thuộc khoảng cách tới camera. Giá trị được kẹp trong [−8, 8]. Nếu không thấy vai thì dùng tâm (0,5; 0,5) và thước đo 0,25. **Scaler có mặt nạ**: `StandardScaler` thông thường bị các điểm thiếu (bằng 0) kéo lệch trung bình và độ lệch chuẩn, nên em tính mean/std chỉ trên các khối có dữ liệu, chỉ fit trên train, và giữ điểm thiếu bằng 0 sau chuẩn hóa (kiểm tra trên val: mean 0,011, std 0,995).


### 2.3. Kiến trúc mô hình

```
Input (30, 204)
  → BiLSTM(128, return_sequences) → Dropout 0,3
  → BiLSTM(64,  return_sequences)
  → TemporalAttention(64)         # attention cộng kiểu Bahdanau trên trục thời gian
  → Dropout 0,5 → Dense(128, ReLU) → Dropout 0,4
  → Dense(100, softmax)
```

| Lớp | Đầu ra | Số tham số |
|---|---|---:|
| BiLSTM 1 | (30, 256) | 340.992 |
| BiLSTM 2 | (30, 128) | 164.352 |
| Temporal attention | (128) | 8.321 |
| Dense | (128) | 16.512 |
| Softmax | (100) | 12.900 |
| **Tổng** | | **543.077** |

Lớp BiLSTM cho phép mỗi bước thời gian "nhìn" cả quá khứ lẫn tương lai của cử động, điều hữu ích khi hai ký hiệu giống nhau ở nửa đầu nhưng khác ở nửa sau. Lớp attention học một trọng số cho từng khung rồi lấy trung bình có trọng số, nhờ đó mô hình tập trung vào các khung mang thông tin (lúc tay đạt tư thế chính) thay vì phải nén toàn bộ thông tin vào bước cuối. Em chưa làm thí nghiệm bỏ attention hay bỏ khối mặt, nên các lựa chọn này mới dựa trên lập luận chứ chưa có bằng chứng so sánh (mục 5).

### 2.4. Huấn luyện

| Hạng mục | Thiết lập |
|---|---|
| Tối ưu | AdamW, lr 5×10⁻⁴, weight decay 10⁻⁴; batch 32 (60 batch/epoch) |
| Điều chuẩn | Dropout 0,2 (trong LSTM), 0,3 / 0,5 / 0,4; L2 10⁻⁴ |
| Hàm mất | Cross-entropy; trọng số lớp = (số mẫu TB / số mẫu lớp)^0,5, cân bằng nhẹ |
| Lịch lr | Giảm một nửa nếu val_loss không cải thiện sau 6 epoch (sàn 10⁻⁶) |
| Dừng sớm | Theo val_loss, patience 20, min_delta 0,001, khôi phục trọng số tốt nhất |
| Ngẫu nhiên | seed 42 (augment xác định theo seed, epoch, chỉ số mẫu) |
| Môi trường | TensorFlow 2.21.0, Keras 3.13.2, MediaPipe 0.10.32, scikit-learn 1.6.1; GPU Colab |

**Tăng cường dữ liệu.** Mỗi mẫu train được augment với xác suất 0,8; 20% còn lại giữ nguyên để phân phối huấn luyện không lệch khỏi dữ liệu thật. Phép biến đổi làm trên tọa độ đã chuẩn hóa theo vai và luôn giữ điểm thiếu bằng 0:

| Nhóm | Chi tiết |
|---|---|
| Thời gian | cắt 75–100% độ dài; một nửa số lần đổi nhịp phi tuyến (tốc độ 0,7–1,3); nội suy chỉ khi cả hai khung lân cận có dữ liệu |
| Không gian | xoay ±10°, co giãn 0,9–1,1, shear ±0,08, tịnh tiến ±0,1 |
| Nhiễu | Gaussian σ = 0,01 trên điểm có mặt |
| Mất tay | xác suất 0,15 mỗi tay, che 2–7 khung (mô phỏng MediaPipe mất dấu) |

Phép "mất tay" nhắm thẳng vào điểm yếu đo được ở mục 2.1; mục 4.3 cho thấy nó mới giảm nhẹ chứ chưa xử lý được vấn đề.

## 3. Thiết kế đánh giá

Em tách vai trò ba tập rõ ràng: train để học, validation để chọn checkpoint và dừng sớm, test chỉ để đo cuối. Mọi quyết định trong quá trình huấn luyện (epoch tốt nhất, lịch giảm lr, dừng sớm) dựa trên val_loss, không dùng test. Chỉ số báo cáo: accuracy Top-1 và Top-5; precision/recall/F1 trung bình macro (mọi lớp như nhau) và weighted (theo số mẫu lớp); khoảng tin cậy Wilson 95% cho accuracy; sai số hiệu chuẩn kỳ vọng (ECE, 10 khoảng) cho độ tin cậy. Vì nhiều lớp chỉ có 1–2 clip test nên chỉ số theo lớp rất nhiễu; em ưu tiên accuracy và weighted-F1 làm số tổng hợp, và xem các phân tích theo nhóm (độ tin cậy, chất lượng landmark, cỡ lớp) là bằng chứng chính cho nguyên nhân lỗi.

## 4. Kết quả và phân tích

### 4.1. Quá trình huấn luyện

![Đường cong huấn luyện]

Mô hình chạy 83 epoch và dừng sớm. Epoch tốt nhất theo val_loss là **63** (val_loss 1,0426; val accuracy 79,43%; train accuracy 86,67%), đúng bằng 83 trừ patience 20. Learning rate giảm 5 lần, ở epoch 44, 50, 69, 75 và 81 (từ 5×10⁻⁴ xuống 1,56×10⁻⁵). Sau epoch 63, loss train tiếp tục giảm trong khi val_loss đi ngang (khoảng 1,05), dấu hiệu bắt đầu overfit nhẹ; dừng sớm và khôi phục trọng số epoch 63 xử lý đúng chỗ này. Em báo cáo val accuracy **tại epoch được chọn** (79,43%), không lấy giá trị cao nhất qua các epoch (80,86% ở epoch 78), vì chọn theo giá trị cao nhất trên val sẽ làm số liệu lạc quan. Cũng cần nhớ rằng đường train trong Hình 1 đo khi augment và dropout đang bật nên không so trực tiếp được với đường validation.

### 4.2. Kết quả trên tập test

| Chỉ số | File kết quả CSV (n = 418) | Log notebook / `config.json` |
|---|---:|---:|
| Accuracy Top-1 | **83,97%** (351/418) | 84,21% (352/418) |
| Khoảng tin cậy Wilson 95% | 80,1% – 87,2% | 80,4% – 87,4% |
| Top-5 | n/a (file CSV không có) | 95,45% |
| Macro precision / recall / F1 | 77,27% / 76,58% / 75,07% | 75,48% / 75,86% / 73,89% |
| Weighted precision / recall / F1 | 84,49% / 83,97% / 82,86% | 84,44% / 84,21% / 83,01% |

Hai cột khác nhau chưa tới 0,3 điểm ở accuracy, nằm gọn trong khoảng tin cậy của nhau, nên kết luận chung không đổi: **mô hình đúng khoảng 84% (khoảng 80–87%) số clip ký hiệu đơn ở 100 lớp**. Chênh lệch ở macro-F1 (khoảng 1,2 điểm) lớn hơn vì chỉ cần vài clip đổi dự đoán ở các lớp 1 clip là F1 của lớp nhảy từ 0 lên 1. Val accuracy tại epoch chọn là 79,43%, thấp hơn test khoảng 4 điểm; hai tập gồm các nhóm video khác nhau và mỗi lớp chỉ có vài nhóm nên độ lệch này nằm trong dao động hợp lý, nhưng em không nên đọc 84% là con số chắc chắn.

### 4.3. Phân tích lỗi

Có 67 clip dự đoán sai, trải trên 64 cặp (nhãn thật → nhãn đoán) khác nhau. Chỉ hai cặp lặp lại (Họ → Đồng ý: 3 lần; Đi → Phía sau: 2 lần), 62 cặp còn lại chỉ xảy ra một lần, nên lỗi **không tập trung vào vài cặp ký hiệu dễ nhầm** mà phân tán. Có 49 lớp đúng hết, 25 lớp đạt F1 = 1, 64 lớp có F1 ≥ 0,8, và 10 lớp không đúng clip nào: Bệnh nhân, Cơ thể, Ho, Khu cách ly, Ngón tay, Sốt, Thương, Virus, Xe máy, Ủng hộ. Cả 10 lớp này chỉ có đúng 1 clip test, nghĩa là "F1 = 0" ở đây là một clip bị đoán sai.

![Độ đúng theo chất lượng phát hiện tay và theo cỡ lớp]

**Chất lượng phát hiện tay .** Khi MediaPipe thấy ít nhất một tay ở ≤50% số khung, độ đúng chỉ là 65,8% (n = 79); ở 50–75% là 82,9% (n = 82); ở >75% là 89,9% (n = 257). Nhóm 79 clip này chiếm 19% số clip test nhưng gây 27 trên 67 lỗi (40%). Với test Mann–Whitney, tỉ lệ khung có tay của các clip dự đoán sai (trung bình 0,61) thấp hơn có ý nghĩa so với clip dự đoán đúng (0,81), p ≈ 9×10⁻⁶. Trong 6 clip test hoàn toàn không thấy tay, 5 clip bị đoán sai. Ngược lại, tỉ lệ thấy tay trung bình của cả lớp (tính trên train) **không** tương quan với F1 của lớp (Spearman ρ = 0,07, p = 0,50). Hai kết quả cùng nói rằng ảnh hưởng nằm ở mức từng clip chứ không phải ở mức lớp: lớp nào cũng có những clip khó, và clip khó là clip mà bước trích đặc trưng làm hỏng. Đây là tương quan, không chứng minh được quan hệ nhân quả (cử động nhanh có thể vừa làm MediaPipe mất dấu vừa làm ký hiệu khó nhận), nhưng nó chỉ ra hướng cải thiện rẻ nhất: cải thiện bước phát hiện tay và bù khung bị mất.

**Cỡ lớp.** Các lớp chỉ có 1–2 clip test (33 lớp) đạt recall gộp 62,8%, nhóm 3–4 clip (15 lớp) đạt 73,1%, nhóm 5–6 clip (32 lớp) đạt 87,6% và nhóm 7–8 clip (20 lớp) đạt 89,7%. Số clip test của một lớp phản ánh số nhóm video gốc của lớp đó, nên đây cũng là bằng chứng gián tiếp rằng lớp ít dữ liệu học kém hơn. Một phần chênh lệch này là nhiễu thống kê vì nhóm đầu rất nhỏ (43 clip). Tuy vậy xu hướng đơn điệu qua bốn nhóm là nhất quán.

### 4.4. Độ tin cậy có đáng tin không

Giao diện web hiển thị Top-3 cùng xác suất, nên câu hỏi "xác suất đó có ý nghĩa không" quan trọng không kém accuracy. Em kiểm tra bằng cách chia 418 dự đoán test theo xác suất của lớp được chọn. Độ tin cậy còn phụ thuộc vào nhiều yếu tố như động tác có thức sự chính xác hay không, cam nhận diện có tốt không,...

![Độ tin cậy và độ đúng thực tế]

| Độ tin cậy | Số clip | Độ đúng |
|---|---:|---:|
| < 0,3 | 10 | 20% |
| 0,3 – 0,5 | 19 | 47% |
| 0,5 – 0,7 | 46 | 41% |
| 0,7 – 0,9 | 53 | 75% |
| 0,9 – 0,99 | 137 | 93% |
| ≥ 0,99 | 153 | 100% |

Độ tin cậy trung bình của các dự đoán đúng là 0,92 và của dự đoán sai là 0,61. Độ tin cậy trung bình toàn bộ (0,873) cao hơn độ đúng (0,840) một chút, tức mô hình hơi quá tự tin, và ECE bằng 0,044. Ở mức cao (≥0,9) độ tin cậy gần như phản ánh đúng xác suất đúng. Ở khoảng giữa (0,5–0,7) thì không: độ đúng chỉ khoảng 41% dù xác suất ghi 0,5–0,7. Em khuyến nghị dùng độ tin cậy như một bộ lọc chứ không đọc nó như xác suất chính xác:

| Ngưỡng độ tin cậy | Tỉ lệ clip còn lại | Độ đúng trên các clip đó | Số lỗi còn lại |
|---|---:|---:|---:|
| 0 (không lọc) | 100% | 84,0% | 67 |
| 0,5 (mặc định của web) | 93,1% | 87,4% | 49 |
| 0,7 | 82,1% | 93,6% | 22 |
| 0,9 | 69,4% | 96,9% | 9 |

Vẫn có 9 lỗi có độ tin cậy ≥0,9 (ví dụ nhầm "Cảm ơn" thành "Áp dụng" với 0,98, "Ủng hộ" thành "Hy sinh" với 0,98), nên không có ngưỡng nào loại hết lỗi. Con số này đo trên clip đơn đã cắt sẵn của bộ dữ liệu; trong thực tế webcam (ánh sáng, tốc độ khung, người ký khác) hiệu chuẩn có thể xấu hơn.

## 5. Hệ thống ứng dụng (VSL Live)

Em đóng gói mô hình thành web gồm backend FastAPI, cơ sở dữ liệu SQLite và giao diện JavaScript thuần do FastAPI phục vụ. Luồng webcam: trình duyệt gửi từng khung JPEG qua WebSocket (chỉ gửi khung kế tiếp sau khi nhận phản hồi khung trước để không dồn hàng đợi); server trích landmark, tách đoạn theo chuyển động cổ tay/khuỷu, lấy mẫu đều 30 khung, chuẩn hóa theo vai, qua scaler rồi vào mô hình và trả Top-3. Luồng video: tải lên, xử lý ở nền, tách đoạn theo chênh lệch ảnh xám (ngưỡng chuyển động 0,012, nghỉ 0,65 giây, đoạn 0,35–4,5 giây). Chuỗi nhãn được Groq (`openai/gpt-oss-20b`, temperature 0) diễn đạt thành câu; khi Groq lỗi hoặc chưa có khóa, web ghép nhãn theo quy tắc và hiển thị rõ đó là bản dự phòng. Web còn có từ điển 100 ký hiệu, lịch sử (tìm kiếm, xuất CSV/JSON), cài đặt và trang trạng thái hệ thống (nhật ký lỗi, hiệu năng).

**Các lần chạy thực tế**: webcam, một đoạn 1,26 giây gồm 20 khung ở 15,1 khung/giây (640×480), kết quả Cảm ơn 96,3% (Top-3: Nói xấu 3,4%, Ăn mừng 0,1%); video 3,27 giây ra một đoạn "Bệnh nhân" 98,5%, xử lý 6,5 giây gồm lần nạp đầu của mô hình. Hai lần thử này cho thấy đường ống chạy thông từ đầu đến cuối, nhưng em không có nhãn thật cho chúng nên không tính thành độ chính xác. Đáng chú ý là ở lần thử webcam, nhãn tay MediaPipe đổi giữa trái và phải 3 lần trong một đoạn ngắn và tốc độ chỉ khoảng 15 khung/giây (một nửa so với 30 khung khi huấn luyện); web hiển thị các thông số này ở mục chẩn đoán để người dùng thấy.

Kiểm thử tự động gồm 32 test đạt và 1 bỏ qua; các test dùng bản giả của MediaPipe và TensorFlow nên chỉ kiểm tra đường ống (tách đoạn, phiên webcam, job video, dịch Groq với mạng giả lập, API, cài đặt) và **không đo độ chính xác**.

## 6. Thảo luận: các mối đe dọa tính hợp lệ và hạn chế

1. **Có thể rò rỉ theo người ký.** Bộ dữ liệu không cung cấp ID người ký, nên một người ký có thể xuất hiện ở cả train và test. Khi đó mô hình học một phần đặc điểm cá nhân và điểm test lạc quan hơn so với người lạ. Em không kiểm tra được điều này với dữ liệu hiện có. Ngoài ra, nhóm tách theo tiền tố tên file trước dấu `_`; nhiều file không có dấu này nên mỗi file là một nhóm riêng, tức việc tách "theo nhóm" thực chất yếu hơn tên gọi của nó.
2. **Bản sao ngầm.** Băm file bắt được 550 nhóm trùng, nhưng cặp `Xúc động` ở mục 2.1 cho thấy còn các bản mã hóa lại không bắt được bằng băm file. Kiểm tra ở mức landmark chỉ bắt được các cặp giống hệt.
3. **Tập test nhỏ.** Mỗi lớp trung bình 4,2 clip test (nhỏ nhất 1). Bất kỳ số liệu theo lớp nào cũng nhiễu; khoảng tin cậy của accuracy tổng rộng khoảng 7 điểm.
4. **Chưa có ablation.** Em chưa so sánh bản có và không có khối mặt, có và không có attention, các cỡ LSTM khác nhau hay từng loại augment, và mới chạy một seed. Do đó chưa khẳng định được đóng góp riêng của từng thành phần.
5. **Ghép chuỗi và dịch câu chưa được đo.** Em chưa có bộ video nhiều ký hiệu có nhãn nên chưa đo được tỉ lệ cắt đúng ranh giới, đoạn bị bỏ sót hay đếm lặp, cũng như chất lượng câu Groq (LLM có thể thêm hoặc bớt từ khi diễn đạt gloss tiếng Việt vốn thiếu ngữ pháp). Mô hình cũng không có lớp "không ký / chuyển tiếp", nên luôn trả về một trong 100 nhãn; chỉ có ngưỡng độ tin cậy để lọc.
6. **Khoảng cách giữa dữ liệu và webcam.** Bộ dữ liệu là clip đã cắt sẵn; webcam trực tiếp có tốc độ khung thấp hơn, nhãn tay có thể đổi, và người ký lạ. Số liệu mục 4 vì thế là cận trên hợp lý cho điều kiện thực tế.

## 7. Kết luận và hướng phát triển

Hệ thống đạt khoảng 84% accuracy Top-1 (80–87%) và 95% Top-5 trên 100 ký hiệu đơn với mô hình chỉ 543 nghìn tham số, chạy được trên máy cá nhân. Điểm đáng tin nhất của kết quả là độ tin cậy có hiệu chuẩn khá tốt (ECE 0,044), cho phép web từ chối bớt các dự đoán yếu: ngưỡng 0,7 giữ 82% số clip với độ đúng 93,6%. Điểm yếu rõ nhất là phụ thuộc vào chất lượng phát hiện tay và vào lượng dữ liệu mỗi lớp.

Hướng tiếp theo em sẽ làm:

1. Thu hoặc quay một bộ video nhiều ký hiệu có nhãn ranh giới để đo bộ tách đoạn và chất lượng câu dịch.
2. Cải thiện bước phát hiện tay: thử bù khung bị mất bằng nội suy hoặc đặc trưng "tay có mặt" làm mặt nạ đầu vào, và thử MediaPipe ở ngưỡng phát hiện thấp hơn. Bước này nhắm trực tiếp vào nhóm lỗi lớn nhất ở mục 4.3.
3. Thêm lớp "không ký / chuyển tiếp" và huấn luyện lại, để mô hình có thể từ chối.
4. Làm ablation (bỏ mặt, bỏ attention, đổi cỡ LSTM, tắt từng augment) với nhiều seed, và hiệu chuẩn lại xác suất (temperature scaling) trên val.
5. Thu thêm clip cho các lớp có ít dữ liệu; kiểm tra rò rỉ theo người ký nếu có thể lấy thông tin này từ nguồn dữ liệu.

## Phụ lục. Danh sách tệp

| Tệp | Nội dung |
|---|---|
| `figs/hinh1_huan_luyen.png` | Đường cong huấn luyện (tái tạo từ log notebook, 83 epoch; số liệu ở `figs/history_from_log.csv`) |
| `figs/hinh2_do_tin_cay.png` | Hiệu chuẩn độ tin cậy và đánh đổi độ phủ–độ đúng (từ `test_predictions.csv`) |
| `figs/hinh3_nguyen_nhan_loi.png` | Độ đúng theo chất lượng phát hiện tay (từ `extraction_qc.csv`) và theo cỡ lớp (từ `classification_report.csv`) |

## Tài liệu tham khảo

- Google. MediaPipe Solutions: Hand Landmarker, Pose Landmarker, Face Landmarker (tài liệu MediaPipe Tasks).
- Bộ dữ liệu: `namwu/ViSignLanguage-Video`, Hugging Face (theo notebook).
- Hochreiter, S., Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.
- Schuster, M., Paliwal, K. K. Bidirectional Recurrent Neural Networks. IEEE Trans. Signal Processing, 1997.
- Bahdanau, D., Cho, K., Bengio, Y. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR, 2015.
- Loshchilov, I., Hutter, F. Decoupled Weight Decay Regularization. ICLR, 2019.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. On Calibration of Modern Neural Networks. ICML, 2017.
- Wilson, E. B. Probable Inference, the Law of Succession, and Statistical Inference. JASA, 1927.