# KỊCH BẢN THUYẾT TRÌNH — BẢN CÓ GIẢI THÍCH LOGIC

**Mục đích file này:** giữ nguyên lời thoại của bản gốc, nhưng sau mỗi đoạn có thêm chú thích `💡 LOGIC` để giải thích *tại sao* nói như vậy, dụng ý đằng sau, và rủi ro cần tránh.

> File để đọc khi thuyết trình: `KICH_BAN_THUYET_TRINH.md`
> File này để **học và hiểu** trước khi lên trình bày.

---

# SLIDE 1 — Dataset Partition & Model Architecture

### Lời dẫn vào
> "Ở phần này em xin trình bày hai bước nền tảng của hệ thống: cách chúng em chia dữ liệu, và kiến trúc mô hình hai tầng — đây chính là trái tim của cả dự án."

💡 **LOGIC:** Câu mở đầu luôn phải "đóng khung" (framing) cho người nghe biết sắp nghe gì → não họ chuẩn bị sẵn 2 cái ngăn để xếp thông tin. Cụm "trái tim của cả dự án" là một **tín hiệu nhấn mạnh**: báo cho hội đồng rằng slide này đáng tập trung, đừng lướt. Nếu không có câu này, người nghe phải tự đoán cấu trúc và dễ bị lạc.

---

### Khối bên trái — Chia dữ liệu
> "Đầu tiên về dữ liệu. Chúng em dùng bộ RWF-2000, là một trong những benchmark chuẩn và khó nhất cho bài toán phát hiện bạo lực, vì nó gồm các clip CCTV thực tế chứ không phải phim dàn dựng."

💡 **LOGIC:** Phải nói RWF-2000 là "benchmark chuẩn" để **chứng minh tính hợp lệ** — dùng dữ liệu mà cộng đồng công nhận thì kết quả mới được tin. Cụm "khó nhất... CCTV thực tế chứ không phải phim dàn dựng" là để **phòng thủ trước**: nếu accuracy không quá cao (87%), người nghe sẽ tự hiểu là do dataset khó, chứ không phải mô hình kém. Đây là kỹ thuật "hạ kỳ vọng đúng lúc".

> "Tổng cộng có 1.989 clip, mỗi clip 5 giây, 30 hình mỗi giây. Chúng em chia theo tỉ lệ 70-15-15: 1.392 clip để huấn luyện, 298 clip cho validation, và 299 clip để test."

💡 **LOGIC:** Nói số cụ thể để tạo **độ tin cậy** — người trình bày nắm rõ dữ liệu của mình. Việc nêu rõ 3 tập (train/val/test) cho thấy bạn theo đúng quy trình ML chuẩn: train để học, val để chọn mô hình, test để báo cáo. Hội đồng kỹ thuật sẽ kiểm tra điều này.

> "Có hai điểm em muốn nhấn mạnh. Thứ nhất, cách chia là stratified — tỉ lệ violent và normal được giữ cân bằng trong cả ba tập... Thứ hai, chúng em cố định seed 42 và lưu vào file split.json, dùng chung cho tất cả thí nghiệm."

💡 **LOGIC:** Đây là 2 "vũ khí" chống lại 2 câu hỏi phản biện kinh điển:
- **Stratified** chống câu "lỡ tập test toàn clip dễ thì sao?" → cân bằng lớp đảm bảo không bị lệch.
- **Seed cố định + split.json** chống câu "kết quả có phải ăn may một lần không?" → tái lập được (reproducible). Trong nghiên cứu, **reproducibility** là tiêu chí sống còn; nói ra điều này cho thấy bạn làm việc nghiêm túc, không "vẽ" số liệu.

---

### Câu chuyển sang khối phải
> "Sau khi có dữ liệu sạch và chia ổn định, chúng em xây dựng kiến trúc hai tầng. Em xin đi qua từng bước theo sơ đồ bên phải."

💡 **LOGIC:** Câu chuyển (transition) giúp người nghe **không bị hẫng** khi mắt chuyển từ trái sang phải slide. Cụm "dữ liệu sạch và chia ổn định" tóm tắt lại phần vừa nói rồi mới đi tiếp — đây là kỹ thuật "khóa kết luận cũ trước khi mở phần mới".

---

### Khối bên phải — Kiến trúc mô hình
> "Tầng một, ô số 1 — X3D-S, và quan trọng là nó được đóng băng... Sau khi fine-tune xong, chúng em khóa toàn bộ trọng số lại, không huấn luyện thêm."

💡 **LOGIC:** Từ "đóng băng" (frozen) phải nói **2 lần và nhấn mạnh** vì nó là nền tảng cho toàn bộ tính mới của đề tài. Nếu X3D không đóng băng thì CGM không còn là "module gắn thêm" nữa, và mất luôn tính model-agnostic. Người nghe phải hiểu rõ điểm này thì mới hiểu được giá trị của CGM ở các bước sau.

> "Ô số 2 — ba luồng ngữ cảnh... Điểm mấu chốt là cả ba luồng không cần gán nhãn thủ công — hoàn toàn tự động, nên rất rẻ để triển khai."

💡 **LOGIC:** Cụm "không cần gán nhãn thủ công" là một **điểm bán hàng (selling point)** cực mạnh. Trong thực tế, gán nhãn là việc tốn kém nhất của AI. Nói "annotation-free" = "phương pháp của em triển khai được ngoài đời thật mà không tốn tiền thuê người gán nhãn". Đây là cách kết nối nghiên cứu với giá trị thực tiễn — hội đồng rất thích điều này.

> "Ô số 3 — ghép nối. Chúng em gộp p_base với 12 đặc trưng ngữ cảnh thành một vector 13 chiều, rồi chuẩn hóa bằng StandardScaler."

💡 **LOGIC:** Nói "StandardScaler" (chuẩn hóa) để cho thấy bạn xử lý dữ liệu đúng kỹ thuật — các feature có thang đo khác nhau (đếm người vs độ sáng), nếu không chuẩn hóa thì feature có giá trị lớn sẽ lấn át. Một chi tiết nhỏ nhưng cho thấy sự cẩn thận chuyên môn.

> "Ô số 4 — Context Gating Module... Nó chỉ có 962 tham số — cực kỳ nhẹ so với hàng triệu tham số của X3D."

💡 **LOGIC:** Phép so sánh "962 vs hàng triệu" là **kỹ thuật tương phản con số** để người nghe cảm nhận ngay mức độ "nhẹ". Con số 962 đứng một mình thì vô nghĩa; đặt cạnh "hàng triệu" thì lập tức thấy ấn tượng. Đây là cách làm số liệu "biết nói".

---

### Khối công thức
> "Tất cả hội tụ ở công thức: p_final = alpha nhân p_base, cộng với (1 trừ alpha) nhân p_ctx."

💡 **LOGIC:** Phải đọc công thức ra lời (không chỉ chỉ tay vào slide) vì công thức là **bằng chứng đây là nghiên cứu có cơ sở toán học**, không phải làm mò. Nhưng đọc xong phải giải thích ngay (câu sau) — công thức trần trụi sẽ làm người nghe sợ.

> "Cách hiểu trực giác: alpha là mức độ hệ thống tin vào X3D. Khi bối cảnh bình thường, alpha cao... Khi bối cảnh cho thấy đây có thể là báo động giả, alpha giảm..."

💡 **LOGIC:** Đây là câu **quan trọng nhất slide 1**. Sau mỗi công thức toán PHẢI có một câu "trực giác" (intuition) dịch toán sang ngôn ngữ đời thường. "alpha = mức độ tin X3D" biến một biến số khô khan thành một khái niệm con người hiểu được. Hội đồng có thể quên công thức nhưng sẽ nhớ "à, alpha là cái nút vặn độ tin cậy" → đó là mục tiêu. Đây cũng chính là lý do CGM "interpretable" (giải thích được) — một đóng góp của paper.

---

### Câu chốt chuyển slide
> "Thiết kế này có một ưu điểm lớn: vì X3D bị đóng băng và CGM tách rời, nên framework này gắn được vào bất kỳ detector nào chứ không riêng X3D."

💡 **LOGIC:** Câu này "thu hoạch" giá trị của việc đóng băng đã gài ở trên → **model-agnostic**. Đặt ở cuối slide vì đây là điểm muốn người nghe mang theo. Đồng thời nó mở đường cho RQ3 (generalization) ở slide sau — tạo mạch logic liền mạch giữa các slide.

> "Bây giờ em xin chuyển sang phần quan trọng nhất — kết quả thực nghiệm."

💡 **LOGIC:** Báo trước "phần quan trọng nhất" để **kéo sự chú ý** lên cao trước slide 2. Sau khi nói lý thuyết, người nghe luôn chờ câu hỏi "thế nó có chạy được không?" → câu này hứa hẹn sẽ trả lời ngay.

---

# SLIDE 2 — Ablation Study E0 → E5

### Lời dẫn vào
> "Đây là slide kết quả cốt lõi. Chúng em thiết kế một ablation study — tức là thêm từng thành phần một để đo chính xác mỗi phần đóng góp bao nhiêu."

💡 **LOGIC:** Phải định nghĩa "ablation study" ngay vì không phải ai trong hội đồng cũng quen thuật ngữ. Giải thích "thêm từng thành phần một" cho thấy đây là **thiết kế thí nghiệm có kiểm soát** (controlled experiment) — chuẩn khoa học. Nó chứng minh bạn không chỉ khoe kết quả cuối, mà chứng minh được *vì sao* có kết quả đó.

> "Chỉ số em muốn các thầy chú ý nhất là FPR — tỉ lệ báo động giả... mục tiêu là FPR phải giảm, trong khi FNR — tỉ lệ bỏ sót — không được tăng."

💡 **LOGIC:** Phải "hướng dẫn cách đọc bảng" trước khi trình bày bảng. Nếu không, mắt người nghe sẽ chạy lung tung khắp 6 cột. Nói rõ "nhìn FPR, và canh chừng FNR" = đưa cho họ một **cặp kính lọc** để biết đâu là thành công, đâu là cái giá phải trả. Đây là cách kiểm soát sự chú ý của khán giả.

---

### Bốn ô số lớn
> "Ô đỏ: baseline E0... FPR là 0.1533 — nghĩa là cứ 100 clip bình thường thì hệ thống báo nhầm khoảng 15 clip."

💡 **LOGIC:** Phải dịch "0.1533" thành "15 trên 100 clip" vì **số thập phân không gây cảm xúc, còn ví dụ cụ thể thì có**. Người nghe hình dung được "15 lần báo nhầm" là phiền toái thế nào → họ tự thấy bài toán đáng giải. Màu đỏ của ô = tín hiệu "đây là vấn đề".

> "Ô xanh dương: mức giảm tương đối 26.1%. Đây là con số em muốn các thầy nhớ — chúng em cắt được hơn một phần tư số báo động giả."

💡 **LOGIC:** Nói thẳng "con số em muốn các thầy nhớ" = **chỉ định trí nhớ** cho khán giả. Một bài thuyết trình thành công nếu người nghe ra về nhớ đúng 1 con số → bạn chủ động chọn con số đó là 26.1%. "Hơn một phần tư" là cách diễn đạt lại 26% cho dễ cảm nhận hơn (phân số trực quan hơn phần trăm).

> "Ô tím: và làm được điều đó chỉ với 962 tham số thêm vào."

💡 **LOGIC:** Đặt "962 tham số" ngay sau "26.1%" để tạo cặp **hiệu quả cao / chi phí thấp** — đây là lập luận thuyết phục nhất trong kỹ thuật: được nhiều mà tốn ít. Hai con số này đứng cạnh nhau mạnh hơn nhiều so với khi tách rời.

---

### Đi qua bảng ablation
> "E1 — thêm luồng đám đông. Điều thú vị là FPR lại tăng lên 0.16... Chúng em báo cáo trung thực điều này."

💡 **LOGIC:** Đây là **nước đi tâm lý quan trọng**. Chủ động chỉ ra một kết quả XẤU (crowd làm tệ hơn) và gọi nó là "báo cáo trung thực" → xây dựng **uy tín**. Khi hội đồng thấy bạn không giấu kết quả xấu, họ sẽ tin các kết quả tốt của bạn hơn. Nghịch lý: thừa nhận điểm yếu lại làm tăng độ tin cậy tổng thể. Nếu giấu E1 đi, khi bị phát hiện sẽ mất hết uy tín.

> "E2 — thêm luồng ánh sáng. FPR giảm xuống 0.1267 — đây là stream đơn lẻ tốt nhất. Điều này hợp lý: rất nhiều báo động giả xảy ra trong điều kiện ánh sáng kém."

💡 **LOGIC:** Không chỉ nêu số (0.1267) mà còn **giải thích vì sao** (ánh sáng kém → báo động giả). Một kết quả có lời giải thích hợp lý thì đáng tin hơn kết quả "tự nhiên tốt". Nó cho thấy bạn hiểu *cơ chế*, không chỉ đọc số. Hội đồng đánh giá cao việc bạn liên hệ được số liệu với nguyên nhân thực tế.

> "E3 — thêm luồng chuyển động. FPR bằng đúng baseline, 0.1533 — tức một mình nó gần như không thay đổi gì."

💡 **LOGIC:** Nói thẳng motion "không thay đổi gì" thay vì lờ đi. ⚠️ **ĐÂY LÀ ĐIỂM RỦI RO NHẤT của cả bài** — vì motion/synchrony vốn được đặt làm một đóng góp. Cách xử lý: thừa nhận nó yếu khi đứng một mình, rồi "cứu" nó ở phần synergy (E4) ngay sau đó. Đừng khẳng định motion mạnh — số liệu sẽ phản bác bạn ngay tại chỗ.

> "E4 — đây là phương pháp đề xuất, dùng cả ba luồng. FPR xuống thấp nhất, 0.1133, đồng thời accuracy và F1 cũng cao nhất bảng."

💡 **LOGIC:** E4 phải được nói với giọng "cao trào" vì đây là kết luận của cả bảng. Nhấn mạnh "FPR thấp nhất VÀ accuracy/F1 cao nhất" để chặn trước câu hỏi "giảm FPR có phải đánh đổi accuracy không?" → không, được cả hai. Đây là bằng chứng mạnh nhất.

---

### Điểm nhấn synergy
> "Các thầy để ý: ánh sáng đơn lẻ cho 0.1267, chuyển động đơn lẻ cho 0.1533. Nhưng khi ghép cả ba lại, kết quả là 0.1133 — tốt hơn bất kỳ luồng đơn lẻ nào."

💡 **LOGIC:** Đây là lập luận **cứu cho việc dùng cả 3 stream**. Nếu chỉ nhìn từng stream, người ta sẽ hỏi "sao không bỏ crowd và motion, chỉ giữ lighting?". Câu trả lời là synergy: tổng lớn hơn tổng các phần. Cách chứng minh: đặt 3 con số cạnh nhau (0.1267, 0.1533 → 0.1133) để người nghe **tự thấy** 0.1133 nhỏ hơn cả → kết luận tự bật ra, không cần áp đặt. Đây cũng là cách gián tiếp biện hộ cho việc motion (E3) yếu mà vẫn được giữ lại.

---

### Trade-off
> "Ở E4, FNR có tăng rất nhẹ từ 0.1275 lên 0.1342... Đây là một trade-off, nhưng rất nhỏ và chấp nhận được."

💡 **LOGIC:** Chủ động nêu nhược điểm (FNR tăng) trước khi hội đồng kịp hỏi → bạn kiểm soát được câu chuyện thay vì bị động đỡ đòn. Quan trọng: phải **định lượng mức độ nhỏ** ("nửa phần trăm") và **biện minh bằng bối cảnh thực tế** (báo động giả nhiều → người vận hành mệt). Đây là kỹ thuật "thừa nhận có kiểm soát": nêu điểm yếu nhưng kèm ngay lý do tại sao nó không nghiêm trọng.

---

### Câu chốt chuyển slide
> "Slide này trả lời được hai trong ba câu hỏi nghiên cứu... Câu hỏi còn lại — liệu nó có tổng quát hóa sang dữ liệu khác không — em xin trình bày ở slide cuối."

💡 **LOGIC:** Kết nối kết quả với **research questions** để cho thấy bài có cấu trúc chặt, không lan man. Câu "còn 1 câu hỏi chưa trả lời" tạo **sự tò mò (cliffhanger)** dẫn tự nhiên sang slide 3, đồng thời thành thật báo trước rằng phần generalization chưa xong — chuẩn bị tâm lý cho hội đồng.

---

# SLIDE 3 — Current Status & Next Steps

### Lời dẫn vào
> "Slide cuối, em xin tổng kết tiến độ dự án và những bước tiếp theo."

💡 **LOGIC:** Báo hiệu đây là slide tổng kết → người nghe biết bài sắp kết thúc, tập trung lần cuối. Slide trạng thái dự án rất quan trọng với báo cáo môn học/đồ án vì hội đồng cần biết **bạn đang ở đâu trong lộ trình**, không chỉ kết quả.

---

### Các phase đã xong
> "Các ô màu xanh là những phần đã hoàn thành. Phase 2: fine-tune X3D-S, đạt validation F1 là 0.8977..."

💡 **LOGIC:** Dùng màu (xanh = xong) để hội đồng **nắm tiến độ trong 1 giây** mà không cần đọc chữ. Nêu lại vài số liệu chốt (F1=0.8977) để nhắc rằng mỗi phase đều có kết quả đo được, không phải "làm cho có". Cách trình bày theo phase cho thấy dự án có **phương pháp luận rõ ràng**.

> "Như vậy toàn bộ phần lõi của hệ thống đã chạy thông và cho kết quả tích cực."

💡 **LOGIC:** Câu tóm tắt này **trấn an** hội đồng: dù còn việc chưa xong, phần quan trọng nhất (core) đã hoạt động. Đặt câu này trước khi nói về phần chưa xong để người nghe vào phần sau với tâm thế tích cực.

---

### Phase 5 đang làm
> "Ô màu cam là phần chúng em đang thực hiện — Phase 5: đánh giá cross-dataset trên bộ RLVS... áp thẳng lên RLVS mà không huấn luyện lại — gọi là zero-shot transfer."

💡 **LOGIC:** Màu cam = đang làm (khác xanh = xong) → trung thực về trạng thái. Phải giải thích "zero-shot transfer" bằng lời thường ("áp thẳng mà không train lại") vì đây là thuật ngữ chuyên môn. Nêu rõ phần chưa xong cho thấy bạn **biết chính xác còn thiếu gì** — điều này tạo ấn tượng tốt hơn là giả vờ mọi thứ đã hoàn hảo.

> "Nếu FPR trên RLVS cũng giảm, điều đó chứng minh framework của chúng em tổng quát hóa được chứ không chỉ ăn may trên một bộ dữ liệu. Đây là lập luận mạnh nhất cho một bài báo."

💡 **LOGIC:** Giải thích **tại sao Phase 5 đáng làm**, không chỉ "còn việc phải làm". Cụm "không chỉ ăn may trên một bộ dữ liệu" đánh trúng nỗi lo lớn nhất của reviewer: overfitting vào một dataset. Cho thấy bạn hiểu phần còn lại không phải việc vặt, mà là phần nâng giá trị khoa học của cả công trình.

---

### Next Steps
> "Cụ thể ba việc tiếp theo... Một... Hai... Ba, hoàn thiện các phần Methodology, Experiments, Results để nộp hội nghị Scopus Q4."

💡 **LOGIC:** Liệt kê đánh số (1-2-3) để kế hoạch **rõ ràng và khả thi**, không mơ hồ. Nhắc "Scopus Q4" để gắn dự án với **mục tiêu đầu ra cụ thể** — cho thấy đây là nghiên cứu nghiêm túc hướng tới công bố, không dừng ở bài tập.

---

### Câu kết
> "Tóm lại, dự án đã chứng minh được ý tưởng cốt lõi: một module gating cực nhẹ, gắn vào detector có sẵn, giảm đáng kể báo động giả mà gần như không tốn thêm chi phí tính toán."

💡 **LOGIC:** Câu kết phải **gói toàn bộ bài vào 1 câu** mà người nghe có thể nhắc lại cho người khác. Nó nhắc lại 3 điểm mạnh: nhẹ + gắn vào sẵn (model-agnostic) + giảm báo động giả. Đây là "thông điệp mang về nhà" (take-home message). Nếu hội đồng chỉ nhớ 1 câu trong cả bài, bạn muốn đó là câu này.

> "Em xin cảm ơn các thầy, và sẵn sàng nhận câu hỏi."

💡 **LOGIC:** Kết thúc lịch sự + chủ động mời câu hỏi = thể hiện sự tự tin, sẵn sàng bảo vệ. Đừng kết thúc lửng lơ ("ờ, hết rồi ạ") vì sẽ làm yếu ấn tượng cuối.

---

# NGUYÊN TẮC CHUNG ĐẰNG SAU TOÀN BỘ KỊCH BẢN

1. **Mỗi con số phải được "dịch" sang ngôn ngữ con người.** 0.1533 → "15 trên 100 clip". Số liệu tạo độ tin, ví dụ tạo cảm xúc — cần cả hai.

2. **Thừa nhận điểm yếu một cách chủ động luôn làm tăng uy tín.** E1 (crowd tệ hơn), E3 (motion vô dụng), FNR tăng — đều được nêu thẳng. Reviewer tin người trung thực hơn người hoàn hảo.

3. **Sau mỗi công thức/thuật ngữ phải có một câu trực giác.** alpha, ablation, zero-shot... đều được dịch sang lời thường ngay sau đó.

4. **Điểm rủi ro lớn nhất = motion/synchrony.** Số liệu cho thấy nó yếu (E3 ≈ baseline, synchrony ngược giả thuyết). Chiến lược: thừa nhận yếu khi đứng riêng, cứu bằng synergy (E4), hứa kiểm tra lại trên RLVS. TUYỆT ĐỐI không khẳng định mạnh về motion → sẽ bị số liệu của chính mình phản bác.

5. **Mỗi slide kết bằng một câu mở đường cho slide sau.** Tạo mạch liền mạch, người nghe không bị đứt đoạn.
