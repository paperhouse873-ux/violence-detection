# Related Work Analysis — Motivating a Context-Aware, Lightweight Gating Module for False-Alarm Reduction

**Project:** *Context-Aware False Alarm Reduction for Violence Detection in Surveillance Videos Using a Lightweight Gating Module*
**Purpose of this document:** survey the five most recent and closely related works, identify the research gap they collectively leave open, and show why our two-stage, model-agnostic gating approach is a justified next step.

> Note: all summaries below are paraphrased from the cited sources; quantitative claims are reported only where stated by the authors. Content was rephrased for compliance with licensing restrictions.

---

## 1. The five most relevant recent papers

### [P1] CUE-Net — Spatial Cropping + Enhanced UniformerV2 + Efficient Additive Attention (CVPRW 2024)
Senadeera et al. propose a transformer-based detector that crops the spatially salient region of each clip, feeds it through an enhanced UniformerV2 backbone, and applies a modified efficient additive attention to fuse local and global spatio-temporal cues. It reports state-of-the-art accuracy on both **RWF-2000** and **RLVS**.
- **What it optimizes:** raw classification accuracy on benchmark test splits.
- **Relevance to us:** same datasets, same binary fight/non-fight task; the current accuracy ceiling.
- [Source](https://openaccess.thecvf.com/content/CVPR2024W/ABAW/html/Senadeera_CUE-Net_Violence_Detection_Video_Analytics_with_Spatial_Cropping_Enhanced_UniformerV2_CVPRW_2024_paper.html)

### [P2] Dual-Branch VideoMamba with Gated Class-Token Fusion (GCTF) (2025)
A state-space-model (SSM) backbone with two branches — one spatial, one temporal — that are continuously combined through a **gating mechanism** on the class token. It targets an optimal accuracy/efficiency balance for near real-time surveillance.
- **What it optimizes:** accuracy under a tight compute budget.
- **Relevance to us:** the clearest precedent that *gated fusion* is effective for this task — but here the gate is **baked inside** the backbone and trained end-to-end.
- [Source](https://arxiv.org/abs/2506.03162)

### [P3] Vi-SAFE — Spatial-Temporal Framework for Efficient Violence Detection (2025)
A lightweight spatial-temporal pipeline that reports **0.88 accuracy on RWF-2000**, clearly above a TSN-only baseline (**0.77**), while emphasizing efficiency for public-space deployment.
- **What it optimizes:** the efficiency/accuracy trade-off for edge surveillance.
- **Relevance to us:** confirms the deployment framing (lightweight, surveillance) we adopt, on our primary dataset.
- [Source](https://arxiv.org/abs/2509.13210)

### [P4] Context-Aware Encoded Transformer + ST-GCN for Fistfight Detection (Sci. Rep., 2025)
Combines a Context-Aware Encoded Transformer (CAET), which models interactions between people and their environment, with a Spatial-Temporal Graph Convolutional Network over **skeletal data** for temporal action localization.
- **What it optimizes:** fine-grained, context-aware localization of violent actions.
- **Relevance to us:** the strongest existing notion of "context" — but it is **expensive and annotation-dependent** (pose estimation, detection, tracking).
- [Source](https://www.nature.com/articles/s41598-025-12531-4)

### [P5] Explainable Framework — Keyframe Selection + Attention CNN + Grad-CAM++ (Sci. Rep., 2026)
An interpretable pipeline that uses unsupervised keyframe selection and attention-based feature learning, with Grad-CAM++ visual explanations, explicitly framed around **redundancy, transparency, and generalization** in video violence detection.
- **What it optimizes:** interpretability and cross-setting generalization.
- **Relevance to us:** validates that **interpretability** and **generalization** are recognized, open priorities in the field.
- [Source](https://www.nature.com/articles/s41598-026-40977-7)

---

## 2. What they have in common (and what they miss)

| Dimension | P1 CUE-Net | P2 VideoMamba-GCTF | P3 Vi-SAFE | P4 CAET+ST-GCN | P5 Explainable CNN | **Ours** |
|---|---|---|---|---|---|---|
| Primary metric | Accuracy | Acc / efficiency | Acc / efficiency | Localization | Acc / interpretability | **FPR (false alarms)** |
| Explicit false-alarm reduction | No | No | No | No | No | **Yes** |
| Uses scene context | No | No | No | Yes (costly) | No | **Yes (annotation-free)** |
| Gating mechanism | No | Yes (in-backbone) | No | No | No | **Yes (external, model-agnostic)** |
| Extra trainable cost | Full backbone | Full backbone | Full backbone | Full + pose stack | Full backbone | **~10³ params on a frozen backbone** |
| Cross-dataset generalization tested | Partial | Partial | No | No | Emphasized | **Yes (RWF-2000 → RLVS)** |
| Interpretable decision weight | No | No | No | Partial | Yes (visual) | **Yes (gate weight α)** |

Three gaps are consistent across all five works:

1. **The objective is almost always accuracy, not false alarms.** None of the five explicitly measures or optimizes the False Positive Rate, even though false alarms are the dominant operational cost of a deployed surveillance trigger. A detector at 88% accuracy can still flood operators with non-events.

2. **Context is either absent or expensive.** Only P4 models context, and it does so through a heavy, annotation-dependent skeletal/transformer stack. The cheap, always-available scene cues — crowd density, lighting condition, motion coherence — are left unused.

3. **Improvement is achieved by replacing the backbone, not augmenting it.** P1–P5 each train a full, dataset-specific network. P2 shows gating works, but the gate is fused inside the backbone and cannot be reused with another detector. There is no lightweight, **model-agnostic** layer that can be attached to an existing frozen detector.

---

## 3. Why this motivates our paper

Our work is positioned precisely in the gap above. The five papers establish three facts that we build on directly:

- **Gating is a proven fusion primitive** (P2) → we keep the gate but **externalize** it into a tiny module (~10³ parameters) that sits on top of a **frozen** detector, making it model-agnostic rather than backbone-specific.
- **Context carries discriminative signal** (P4) → we extract context **without any annotation** through three off-the-shelf streams (crowd via an object detector, lighting via classical image statistics, motion via optical flow), avoiding P4's pose/tracking overhead.
- **Interpretability and generalization are valued open problems** (P5) → our gate weight α is directly interpretable (how much the system trusts the base detector per clip), and we explicitly test transfer from **RWF-2000 to RLVS**.

This yields the contribution none of the five papers offers: a **two-stage, model-agnostic framework whose explicit goal is reducing false alarms (FPR)** while leaving the underlying detector untouched and adding negligible compute. Where prior work asks *"how do we raise accuracy with a better/larger model?"*, we ask the complementary, deployment-driven question *"given any reasonable detector, how do we cut its false alarms cheaply using free scene context?"*

### Mapping to our three research questions
- **RQ1 — Does X3D-S + Context Gating reduce FPR vs X3D-S alone?** addresses Gap 1 (no one targets FPR).
- **RQ2 — Which context stream contributes most to FPR reduction?** addresses Gap 2 (unused, annotation-free context) and quantifies which cheap cue matters.
- **RQ3 — Does the framework generalize across datasets?** addresses Gap 3 + P5's generalization concern, via zero-shot transfer to RLVS.

---

## 4. References

1. Senadeera et al., *CUE-Net: Violence Detection Video Analytics with Spatial Cropping, Enhanced UniformerV2 and Modified Efficient Additive Attention*, CVPR Workshops, 2024. https://openaccess.thecvf.com/content/CVPR2024W/ABAW/html/Senadeera_CUE-Net_Violence_Detection_Video_Analytics_with_Spatial_Cropping_Enhanced_UniformerV2_CVPRW_2024_paper.html
2. Senadeera et al., *Dual Branch VideoMamba with Gated Class Token Fusion for Violence Detection*, arXiv:2506.03162, 2025. https://arxiv.org/abs/2506.03162
3. *A Spatial-Temporal Framework for Efficient Violence Detection in Public Surveillance (Vi-SAFE)*, arXiv:2509.13210, 2025. https://arxiv.org/abs/2509.13210
4. *Automated violence monitoring system for real-time fistfight detection using deep learning-based temporal action localization (CAET + ST-GCN)*, Scientific Reports, 2025. https://www.nature.com/articles/s41598-025-12531-4
5. *An explainable deep learning framework for video violence detection using unsupervised keyframe selection and attention-based CNN*, Scientific Reports, 2026. https://www.nature.com/articles/s41598-026-40977-7

**Supporting / dataset references**
- Cheng et al., *RWF-2000: An Open Large Scale Video Database for Violence Detection*, ICPR, 2021. https://arxiv.org/abs/1911.05913
- Vijeikis et al., *Efficient Violence Detection in Surveillance* (MobileNetV2 + LSTM, ~0.82 acc on RWF-2000), Sensors, 2022. https://www.mdpi.com/1424-8220/22/6/2216
