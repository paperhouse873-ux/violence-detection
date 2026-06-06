# PROJECT SUMMARY — Context-Aware Violence Detection
# Dùng cho session chat mới để Claude hiểu context ngay

---

## 1. THÔNG TIN NHÓM

| | |
|---|---|
| **Trường** | FPT University, Ho Chi Minh City Campus |
| **Mục tiêu** | Paper Scopus Q4 conference |
| **Corresponding author** | Hà Việt Hưng (SE201122) — hungha.060963@gmail.com — ORCID: 0009-0005-9924-4498 |
| **Thành viên** | Nguyễn Việt Nhân (SE201082), Nguyễn Thái Kiệt (SE200734), Trần Bảo Nguyên (SE201012) |
| **GPU local** | RTX 4060 Laptop 8GB VRAM |
| **Môi trường** | Miniconda tại `C:\miniconda3`, env `violence_det`, Python 3.11 |
| **Thư mục dự án** | `C:\Users\HA VIET HUNG\Videos\archive\` |
| **GitHub** | Private repo `violence-detection` |

---

## 2. TIÊU ĐỀ PAPER

**"Context-Aware False Alarm Reduction for Violence Detection in Surveillance Videos Using a Lightweight Gating Module"**

Template: Springer Nature `sn-jnl` (Overleaf: myxmhdsbzkyd)

---

## 3. DATASET

- **RWF-2000**: 1,989 clips (789 train/fight + 800 train/nonFight + 200 val/fight + 200 val/nonFight)
- Split 70/15/15 stratified seed=42: **1,392 train / 298 val / 299 test**
- Tất cả clips: 5 giây, 30fps, 150 frames, resolution đa dạng
- Thiếu 11 fight clips so với chuẩn gốc 2,000 — không ảnh hưởng
- File `split.json` đã tạo, dùng chung cho TẤT CẢ experiments

---

## 4. KIẾN TRÚC HỆ THỐNG

### Pipeline 2-tầng (Model-Agnostic Framework):

**Tầng 1 (frozen):** X3D-S pretrained Kinetics-400, fine-tuned RWF-2000
→ output: `p_base` ∈ [0,1] (violence probability)

**Tầng 2 (trainable, 962 params):** Context Gating Module (CGM)
- Input: 13-dim vector = [p_base | z_crowd(4) | z_light(4) | z_motion(4)]
- MLP-gate → α ∈ [0,1] (mức tin X3D-S)
- MLP-ctx → p_ctx (xác suất hiệu chỉnh từ context)
- **Formula:** `p_final = α · p_base + (1 − α) · p_ctx`
- Decision: Violent if p_final ≥ threshold

### 3 Context Streams (annotation-free):
- **Crowd:** YOLOv8n → mean_count, max_count, count_variance, density_area
- **Lighting:** OpenCV → mean_brightness, contrast_std, blur_score, low_light_ratio
- **Motion:** Farneback optical flow → motion_mean, motion_peak, direction_entropy, synchrony★

---

## 5. KẾT QUẢ THỰC NGHIỆM (ĐÃ CÓ)

### Phase 2 — X3D-S Fine-tuning (RTX 4060, 5 epochs):
- Best checkpoint: **Epoch 5**, Val F1=0.8977, Val FPR=0.1267
- File: `checkpoints/x3ds_best.pth` (36MB)

### Phase 3 — Context Extraction (1,989 clips):
- p_base: violent=0.8836, normal=0.1262, gap=0.757 ✓ Excellent
- motion_synchrony: violent=0.130 > normal=0.110 (ngược lý thuyết — do CCTV noise)
- Files: `cache/p_base.npy`, `cache/z_crowd.npy`, `cache/z_light.npy`, `cache/z_motion.npy`
- `cache/context_13dim.npy` (normalized 13-dim), `cache/scaler.pkl`, `cache/labels.npy`, `cache/splits.npy`

### Phase 4 — Ablation Study E0–E5 (PRIMARY RESULTS):

| Exp | Config | Acc | F1 | FPR | FNR |
|---|---|---|---|---|---|
| **E0** | X3D-S only | 0.8595 | 0.8609 | **0.1533** | 0.1275 |
| E1 | + crowd | 0.8696 | 0.8730 | 0.1600 ↑ | 0.1007 |
| E2 | + lighting | 0.8662 | 0.8649 | 0.1267 ↓ | 0.1409 |
| E3 | + motion | 0.8696 | 0.8721 | 0.1533 = | 0.1074 |
| **E4** | full CGM | 0.8763 | 0.8746 | **0.1133** ↓↓ | 0.1342 |
| E5 | +pos_weight=3 | 0.8696 | 0.8704 | 0.1400 | 0.1208 |

**FPR improvement E0→E4: 0.1533→0.1133, Δ=0.040, 26.1% relative reduction**

Key findings:
- Lighting stream = most important single feature
- Synergy effect: E4 tốt hơn bất kỳ single stream
- FNR tăng nhỏ (+0.0067) — trade-off chấp nhận được
- α analysis: CGM học trust X3D-S ít hơn với non-violent clips
- Files: `results/ablation_results.json`, `results/ablation_table.csv`

---

## 6. PAPER STATUS

### Đã viết:
- Title, Authors (4 thành viên, ORCID, FPT Uni)
- Abstract (~220 từ)
- Introduction (~350 từ, 4 contributions, critique 4 related works)

### Chưa viết:
- Related Work
- Methodology
- Experiments
- Results & Discussion
- Conclusion

### LaTeX setup (Overleaf):
- File: `sn-article.tex` (template Springer Nature sn-jnl)
- References: `references.bib` (6 entries: RWF-2000, X3D, Vijeikis, Lopez, ResnetCrowd, Islam Survey)
- ORCID fix: dùng `\newcommand{\orcidA}{}` thay vì `\orcid{}` (tránh Orcidlogo.eps error)

### 4 Contributions trong paper:
1. Two-stage model-agnostic framework for FPR reduction
2. Annotation-free multi-stream context extraction
3. Formalisation of motion synchrony as discriminative feature
4. CGM với interpretable α attention weight

---

## 7. PHASE 5 — CHƯA HOÀN THÀNH (VẤN ĐỀ)

Yêu cầu thầy: train ≥4 models, so sánh với baseline paper.

**Đã thử và FAIL:**
- SwinV2-S + LSTM: oscillate FPR=1.0↔0.0 (per-frame 2D+LSTM không hội tụ)
- ConvNeXt-S + LSTM: cùng vấn đề
- EfficientNetV2-S + LSTM: cùng vấn đề
- TimeSformer (HuggingFace): oscillate
- VideoMAE (HuggingFace): oscillate
- X3D-M (torch.hub): FPR oscillate, loss stuck 0.6932
- SlowFast-R50, Slow-R50: chưa test đủ

**Nguyên nhân thất bại:**
- Per-frame 2D CNN + LSTM: features ImageNet không violence-specific, gradient vanish
- HuggingFace models: API phức tạp, double activation issue
- torch.hub X3D-M vs pytorchvideo.models.hub X3D-S: khác nhau về loading

**Approach đề xuất thay thế (CHƯA IMPLEMENT):**
Cross-dataset evaluation: Train X3D-S+CGM trên RWF-2000, test trên RLVS dataset
- RLVS đã có sẵn trong Kaggle dataset (có train/Fight, train/NonFight, val/Fight, val/NonFight)
- Zero-shot transfer: CGM trained trên RWF-2000 → evaluate trên RLVS
- Argument: "CGM generalizes across different video distributions"
- Cần viết: `phase6_cross_dataset.py`

---

## 8. FILES QUAN TRỌNG (LOCAL)

```
C:\Users\HA VIET HUNG\Videos\archive\
├── split.json                    ← QUAN TRỌNG: dùng cho tất cả experiments
├── phase0_step1_check_structure.py
├── phase0_step2_integrity.py
├── phase0_step3_statistics.py
├── phase0_step4_split.py
├── phase1_dataset.py             ← Dataset class, đã fix IndexError
├── phase2_finetune_x3ds.py       ← Fine-tune X3D-S
├── phase3_extract_context.py     ← 3 context streams
├── phase4_train_cgm.py           ← CGM + ablation E0-E5
├── phase5_mvit_swin.py           ← Phase 5 (chưa hoàn thành)
├── checkpoints/
│   └── x3ds_best.pth             ← X3D-S model (Epoch 5, F1=0.8977)
├── cache/
│   ├── p_base.npy                ← X3D-S predictions (1989,)
│   ├── z_crowd.npy               ← Crowd features (1989, 4)
│   ├── z_light.npy               ← Lighting features (1989, 4)
│   ├── z_motion.npy              ← Motion features (1989, 4)
│   ├── context_13dim.npy         ← 13-dim normalized (1989, 13)
│   ├── scaler.pkl                ← StandardScaler
│   ├── labels.npy                ← Ground truth (1989,)
│   └── splits.npy                ← 0=train,1=val,2=test (1989,)
└── results/
    ├── ablation_results.json     ← E0-E5 full results
    └── ablation_table.csv        ← Bảng so sánh
```

---

## 9. TECHNICAL NOTES QUAN TRỌNG

### Fix đã áp dụng cho X3D-S:
```python
# PHẢI có dòng này — xóa Softmax bên trong head
model.blocks[-1].proj = nn.Linear(in_features, 1)
model.blocks[-1].act = nn.Identity()  # ← CRITICAL FIX

# PHẢI dùng BCEWithLogitsLoss, KHÔNG dùng BCELoss
criterion = nn.BCEWithLogitsLoss()

# KHÔNG apply sigmoid trước loss
logits = model(videos).squeeze(1)
loss = criterion(logits, labels)  # ← KHÔNG: criterion(sigmoid(logits), labels)
probs = torch.sigmoid(logits.detach())  # chỉ dùng cho metrics
```

### DataLoader trên Windows:
```python
DataLoader(..., num_workers=0, pin_memory=False)  # Windows: num_workers phải 0
```

### Conda environment:
```cmd
C:\miniconda3\Scripts\activate.bat violence_det
```

### Dataset structure sau khi download:
```
RWF-2000/
├── train/
│   ├── fight/      (789 clips .avi)
│   └── nonFight/   (800 clips .avi)
└── val/
    ├── fight/      (200 clips .avi)
    └── nonFight/   (200 clips .avi)
```

---

## 10. VIỆC CẦN LÀM TIẾP THEO

**Ưu tiên 1 (cao):** Phase 6 — Cross-dataset evaluation trên RLVS
- Download RLVS từ Kaggle: `kaggle datasets download -d magicearth25/video-violence-detection-dataset`
- Viết `phase6_cross_dataset.py`:
  - Load X3D-S checkpoint
  - Extract p_base từ RLVS
  - Extract 3 context streams từ RLVS  
  - Apply CGM trained trên RWF-2000 (zero-shot)
  - Report FPR before/after CGM trên RLVS

**Ưu tiên 2 (cao):** Viết paper sections còn lại
- Related Work (~1.5 trang)
- Methodology (~3.5 trang) — pipeline, CGM architecture, training protocol
- Experiments (~1.5 trang) — dataset, ablation design, metrics
- Results & Discussion (~2.5 trang) — tables, analysis, feature importance
- Conclusion (~0.5 trang)

**Ưu tiên 3 (nếu cần):** Phase 5 đúng cách
- Dùng `pytorchvideo.models.hub`: x3d_xs, x3d_m, x3d_l
- Cùng approach với X3D-S (đã proven)
- Làm trên cloud GPU nếu cần

---

## 11. SO SÁNH VỚI LITERATURE

| Model | Acc | FPR | Source |
|---|---|---|---|
| Flow Gated Network† | 86.75% | N/A | Cheng et al. ICPR 2021 |
| MobileNetV2+LSTM† | 82.00% | N/A | Vijeikis et al. Sensors 2022 |
| X3D-S (ours, E0) | 85.95% | 0.1533 | Phase 2 |
| X3D-S + CGM (ours, E4) | **87.63%** | **0.1133** | Phase 4 |

† = taken from published papers, no CGM applied

**Key claim:** X3D-S + CGM (87.63%) surpasses Flow Gated Network (86.75%) while additionally reducing FPR by 26.1%.

---

## 12. REFERENCES (BibTeX keys)

- `cheng2021rwf` — RWF-2000 dataset paper (ICPR 2021)
- `fan2020x3d` — X3D paper (CVPR 2020)
- `vijeikis2022efficient` — MobileNetV2+LSTM (Sensors 2022)
- `lopez2023twostage` — Two-stage pipeline
- `marsden2017resnetcrowd` — ResnetCrowd
- `islam2023survey` — ACM Survey 200+ papers
