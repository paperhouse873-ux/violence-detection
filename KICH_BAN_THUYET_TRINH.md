# KỊCH BẢN THUYẾT TRÌNH — 3 SLIDE

**Dự án:** Context-Aware False Alarm Reduction for Violence Detection
**Cách dùng:** Phần "Lời nói" có thể đọc gần như nguyên văn. Chỗ in nghiêng là ghi chú cho người trình bày (không đọc lên).

**Tổng thời lượng 3 slide:** ~7 phút 15 giây

---

## SLIDE 1 — Dataset Partition & Model Architecture
*(Thời lượng: ~2 phút 30 giây)*

### Lời dẫn vào (15 giây)
"Ở phần này em xin trình bày hai bước nền tảng của hệ thống: cách chúng em chia dữ liệu, và kiến trúc mô hình hai tầng — đây chính là trái tim của cả dự án."

### Khối bên trái — Chia dữ liệu (45 giây)
"Đầu tiên về dữ liệu. Chúng em dùng bộ RWF-2000, là một trong những benchmark chuẩn và khó nhất cho bài toán phát hiện bạo lực, vì nó gồm các clip CCTV thực tế chứ không phải phim dàn dựng.

Tổng cộng có 1.989 clip, mỗi clip 5 giây, 30 hình mỗi giây. Chúng em chia theo tỉ lệ 70-15-15: 1.392 clip để huấn luyện, 298 clip cho validation, và 299 clip để test.

Có hai điểm em muốn nhấn mạnh. Thứ nhất, cách chia là **stratified** — tỉ lệ violent và normal được giữ cân bằng trong cả ba tập, các thầy có thể thấy mỗi tập đều xấp xỉ 50-50. Thứ hai, chúng em cố định **seed 42** và lưu vào file `split.json`, dùng chung cho **tất cả** thí nghiệm. Điều này đảm bảo mọi so sánh sau này đều công bằng và **tái lập được** — ai chạy lại cũng ra đúng các con số này."

### Câu chuyển sang khối phải (10 giây)
"Sau khi có dữ liệu sạch và chia ổn định, chúng em xây dựng kiến trúc hai tầng. Em xin đi qua từng bước theo sơ đồ bên phải."

### Khối bên phải — Kiến trúc mô hình (60 giây)
"**Tầng một, ô số 1 — X3D-S, và quan trọng là nó được đóng băng.** Đây là một mạng 3D pretrained trên Kinetics-400, sau đó chúng em fine-tune trên RWF-2000. Nhiệm vụ của nó là nhìn vào clip và đưa ra một con số `p_base` — xác suất clip đó có bạo lực. Sau khi fine-tune xong, chúng em **khóa toàn bộ trọng số** lại, không huấn luyện thêm.

**Ô số 2 — ba luồng ngữ cảnh.** Song song với X3D, chúng em trích xuất thông tin bối cảnh mà X3D không nhìn tới: mật độ đám đông bằng YOLOv8n, điều kiện ánh sáng bằng OpenCV, và đặc trưng chuyển động bằng optical flow Farneback. Điểm mấu chốt là cả ba luồng **không cần gán nhãn thủ công** — hoàn toàn tự động, nên rất rẻ để triển khai.

**Ô số 3 — ghép nối.** Chúng em gộp `p_base` với 12 đặc trưng ngữ cảnh thành một vector 13 chiều, rồi chuẩn hóa bằng StandardScaler.

**Ô số 4 — Context Gating Module, gọi tắt là CGM.** Đây là đóng góp chính. Nó chỉ có **962 tham số** — cực kỳ nhẹ so với hàng triệu tham số của X3D. Bên trong có hai nhánh: MLP-gate sinh ra trọng số alpha, và MLP-ctx sinh ra một xác suất hiệu chỉnh `p_ctx`."

### Khối công thức dưới cùng (20 giây)
"Tất cả hội tụ ở công thức: `p_final = alpha nhân p_base, cộng với (1 trừ alpha) nhân p_ctx`.

Cách hiểu trực giác: **alpha là mức độ hệ thống tin vào X3D**. Khi bối cảnh bình thường, alpha cao, hệ thống nghe theo X3D. Khi bối cảnh cho thấy đây có thể là báo động giả, alpha giảm, hệ thống dựa nhiều hơn vào ngữ cảnh để hiệu chỉnh. Cuối cùng, nếu `p_final` vượt ngưỡng thì kết luận là bạo lực."

### Câu chốt chuyển slide
"Thiết kế này có một ưu điểm lớn: vì X3D bị đóng băng và CGM tách rời, nên framework này **gắn được vào bất kỳ detector nào** chứ không riêng X3D. Bây giờ em xin chuyển sang phần quan trọng nhất — kết quả thực nghiệm."

### *Phòng thủ câu hỏi (ghi chú riêng)*
- *"Tại sao đóng băng X3D mà không train cả hệ thống?"* → Để chứng minh tính model-agnostic, và để CGM nhẹ, train nhanh, không cần GPU mạnh. X3D đã học tốt phần "hành động", CGM chỉ học phần "hiệu chỉnh bối cảnh".
- *"962 tham số có quá ít không?"* → Đúng là rất ít, nhưng đó là chủ đích: nó chỉ học một hàm gating đơn giản trên 13 chiều. Kết quả slide sau cho thấy nó vẫn hiệu quả.
- *"Vì sao 13 chiều?"* → 1 chiều p_base + 4 crowd + 4 lighting + 4 motion.

---

## SLIDE 2 — Evaluation: Ablation Study E0 → E5
*(Thời lượng: ~3 phút — đây là slide quan trọng nhất, nói chậm)*

### Lời dẫn vào (20 giây)
"Đây là slide kết quả cốt lõi. Chúng em thiết kế một **ablation study** — tức là thêm từng thành phần một để đo chính xác mỗi phần đóng góp bao nhiêu. Chỉ số em muốn các thầy chú ý nhất là **FPR — tỉ lệ báo động giả**, ghi ở góc trên bên phải: mục tiêu là FPR phải giảm, trong khi FNR — tỉ lệ bỏ sót — không được tăng."

### Bốn ô số lớn phía trên (40 giây)
"Bốn con số này tóm tắt toàn bộ câu chuyện.

Ô đỏ: baseline E0, chỉ dùng X3D đơn thuần, FPR là **0.1533** — nghĩa là cứ 100 clip bình thường thì hệ thống báo nhầm khoảng 15 clip.

Ô xanh lá: phương pháp đề xuất E4, X3D cộng CGM, FPR giảm còn **0.1133**.

Ô xanh dương: mức giảm tương đối **26.1%**. Đây là con số em muốn các thầy nhớ — chúng em cắt được hơn một phần tư số báo động giả.

Ô tím: và làm được điều đó chỉ với **962 tham số** thêm vào. Tỉ lệ chi phí trên hiệu quả rất tốt."

### Đi qua bảng ablation (80 giây — nói theo từng dòng)
"Bây giờ em đi qua bảng chi tiết để các thầy thấy từng stream đóng góp ra sao.

**E0** — chỉ X3D, đây là vạch xuất phát: FPR 0.1533.

**E1** — thêm luồng đám đông. Điều thú vị là FPR lại **tăng** lên 0.16. Nghĩa là riêng thông tin đám đông không giúp ích, thậm chí gây nhiễu nhẹ. Chúng em báo cáo trung thực điều này.

**E2** — thêm luồng ánh sáng. FPR giảm xuống 0.1267 — đây là **stream đơn lẻ tốt nhất**. Điều này hợp lý: rất nhiều báo động giả xảy ra trong điều kiện ánh sáng kém, và luồng này giúp hệ thống thận trọng hơn.

**E3** — thêm luồng chuyển động. FPR bằng đúng baseline, 0.1533 — tức một mình nó gần như không thay đổi gì.

**E4** — đây là phương pháp đề xuất, dùng cả ba luồng. FPR xuống thấp nhất, **0.1133**, đồng thời accuracy và F1 cũng cao nhất bảng. Em sẽ giải thích điểm đặc biệt của dòng này ngay sau đây.

**E5** — chúng em thử thêm pos_weight để phạt nặng lỗi bỏ sót, FPR là 0.14, không tốt bằng E4 nên không chọn."

### Điểm nhấn về synergy (30 giây)
"Có một phát hiện quan trọng ở dòng E4 mà em muốn nhấn mạnh. Các thầy để ý: ánh sáng đơn lẻ cho FPR 0.1267, chuyển động đơn lẻ cho 0.1533. Nhưng khi ghép **cả ba** lại, kết quả là 0.1133 — **tốt hơn bất kỳ luồng đơn lẻ nào**. Đây là hiệu ứng **synergy**: các luồng bổ sung cho nhau, CGM học được cách kết hợp chúng thông minh hơn là dùng riêng lẻ. Đây chính là lý do chúng em dùng cả ba thay vì chỉ chọn cái tốt nhất."

### Lưu ý về trade-off (20 giây)
"Một điểm về tính trung thực: ở E4, FNR có tăng rất nhẹ từ 0.1275 lên 0.1342 — tức bỏ sót thêm khoảng nửa phần trăm. Đây là một trade-off, nhưng rất nhỏ và chấp nhận được, đổi lại việc giảm 26% báo động giả. Trong giám sát thực tế, báo động giả quá nhiều khiến người vận hành mệt mỏi và bỏ qua cảnh báo, nên việc giảm FPR mang lại giá trị thực tiễn cao."

### Câu chốt chuyển slide
"Tóm lại, slide này trả lời được hai trong ba câu hỏi nghiên cứu: CGM **có** giảm FPR, và ánh sáng là luồng đóng góp nhiều nhất. Câu hỏi còn lại — liệu nó có tổng quát hóa sang dữ liệu khác không — em xin trình bày ở slide cuối."

### *Phòng thủ câu hỏi (ghi chú riêng)*
- *"Vì sao crowd làm tăng FPR?"* → Trên RWF-2000, cảnh đông người xuất hiện ở cả clip bạo lực lẫn bình thường (lễ hội, đám đông), nên một mình nó dễ gây nhầm. Nhưng khi kết hợp với ánh sáng và chuyển động trong E4, CGM học được cách "lọc" nhiễu này.
- *"26% có ý nghĩa thống kê không?"* → Đang củng cố thêm bằng cross-dataset. Trên test set 299 clip, số FP giảm từ 23 xuống 17 clip.
- *"So với SOTA thì sao?"* → X3D+CGM đạt accuracy 87.63%, nhỉnh hơn Flow Gated Network (86.75%), và quan trọng hơn là còn giảm được FPR — điều mà phần lớn paper khác không đo.

---

## SLIDE 3 — Current Status & Next Steps
*(Thời lượng: ~1 phút 45 giây)*

### Lời dẫn vào (15 giây)
"Slide cuối, em xin tổng kết tiến độ dự án và những bước tiếp theo. Dự án được tổ chức thành các phase rõ ràng, và hiện tại chúng em đã hoàn thành phần lớn."

### Đi qua các phase đã xong (40 giây)
"Các ô màu xanh là những phần đã hoàn thành.

Phase 0 và 1: chuẩn bị dữ liệu và DataLoader.

Phase 2: fine-tune X3D-S, đạt validation F1 là 0.8977.

Phase 3: trích xuất ba luồng ngữ cảnh cho toàn bộ gần 2.000 clip.

Phase 4: xây dựng CGM và chạy ablation study — chính là kết quả các thầy vừa xem, giảm 26% FPR.

Như vậy toàn bộ phần lõi của hệ thống đã chạy thông và cho kết quả tích cực."

### Phase 5 đang làm — nói thẳng thắn (35 giây)
"Ô màu cam là phần chúng em **đang thực hiện** — Phase 5: đánh giá cross-dataset trên bộ RLVS.

Ý tưởng là: chúng em lấy mô hình X3D-S cộng CGM đã huấn luyện trên RWF-2000, rồi áp thẳng lên RLVS mà **không huấn luyện lại** — gọi là zero-shot transfer. Nếu FPR trên RLVS cũng giảm, điều đó chứng minh framework của chúng em **tổng quát hóa được** chứ không chỉ ăn may trên một bộ dữ liệu. Đây là lập luận mạnh nhất cho một bài báo, nên chúng em đang ưu tiên hoàn thành."

### Next Steps (25 giây)
"Cụ thể ba việc tiếp theo, như khung dưới cùng:

Một, áp mô hình lên RLVS theo kiểu zero-shot và đo FPR trước-sau khi có CGM.

Hai, nếu FPR trên RLVS cũng giảm, chúng em xác nhận được khả năng tổng quát hóa.

Ba, hoàn thiện các phần Methodology, Experiments, Results để nộp hội nghị Scopus Q4."

### Câu kết toàn bài (15 giây)
"Tóm lại, dự án đã chứng minh được ý tưởng cốt lõi: một module gating cực nhẹ, gắn vào detector có sẵn, giảm đáng kể báo động giả mà gần như không tốn thêm chi phí tính toán. Phần còn lại là củng cố bằng cross-dataset và hoàn thiện bài báo. Em xin cảm ơn các thầy, và sẵn sàng nhận câu hỏi."

### *Phòng thủ câu hỏi (ghi chú riêng)*
- *"Vì sao Phase 5 chưa xong?"* → Ban đầu định train thêm nhiều model lớn để chứng minh model-agnostic, nhưng các model đó khó hội tụ trên GPU laptop. Nên chuyển sang hướng cross-dataset, vừa khả thi hơn vừa cho lập luận mạnh hơn về generalization.
- *"Khi nào nộp được?"* → Sau khi hoàn thành RLVS evaluation và viết xong 4 section còn lại.
- *"Nếu RLVS không giảm FPR thì sao?"* → Câu hỏi tốt. Nếu vậy, chúng em vẫn báo cáo trung thực, và kết quả đó cũng có giá trị: nó cho thấy giới hạn của phương pháp khi gặp domain shift lớn, là một đóng góp khoa học hợp lệ.

---

## LƯU Ý CHUNG KHI TRÌNH BÀY

- **Slide 2 là slide ăn điểm** — dành nhiều thời gian nhất, nói chậm ở bốn ô số lớn và ở điểm synergy.
- Khi nói số liệu, **dừng một nhịp** sau con số quan trọng (26.1%, 0.1133) để thầy kịp ghi nhận.
- Nếu bị hỏi về **motion stream** (E3 không hiệu quả), trả lời thẳng: "Một mình luồng chuyển động không đủ mạnh trên RWF-2000, nhưng nó đóng góp vào hiệu ứng synergy ở E4. Chúng em cũng đang kiểm tra lại giá trị của nó trên RLVS." → Trung thực mà vẫn giữ thế chủ động. **Đừng khẳng định quá lời về motion/synchrony** vì số liệu chưa ủng hộ.
- Giữ giọng bình tĩnh, tự tin. Khi gặp câu khó, thừa nhận giới hạn thay vì bào chữa — hội đồng đánh giá cao sự trung thực khoa học.

---

## BẢNG SỐ LIỆU CẦN NHỚ (học thuộc trước khi lên)

| Chỉ số | Giá trị | Ghi chú |
|---|---|---|
| Tổng số clip | 1.989 | RWF-2000 |
| Chia train/val/test | 1.392 / 298 / 299 | stratified, seed 42 |
| FPR baseline (E0) | 0.1533 | chỉ X3D-S |
| FPR đề xuất (E4) | 0.1133 | X3D-S + CGM |
| Giảm tương đối | 26.1% | con số quan trọng nhất |
| Số FP giảm | 23 → 17 clip | trên test set |
| Tham số CGM | 962 | cực nhẹ |
| Stream đơn tốt nhất | Lighting (E2) | FPR 0.1267 |
| Accuracy E4 | 87.63% | nhỉnh hơn Flow Gated Network 86.75% |
| Val F1 (X3D-S) | 0.8977 | Phase 2 |
