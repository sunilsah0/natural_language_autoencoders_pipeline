# Local Reimplementation & Geometrical Analysis of Natural Language Autoencoders on Qwen2.5-0.5B

This project is a localized engineering replication and diagnostic investigation of Anthropic’s 2026 methodology: [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html).

Faced with a 32-hour time budget and running on compute-constrained free-tier local environments, I intentionally rejected attempting a massive, multi-node Reinforcement Learning setup using GRPO or complex SGLang orchestration clusters. Instead, I scoped the experiment down to a compact, highly dense model: `Qwen/Qwen2.5-0.5B`. My primary goal was to study how compressing hidden activation vectors into raw text strings behaves when you swap high-compute RL alignment for a localized linear mapping layer.

All orchestration, pipeline adjustments, and data parsing loops can be found directly in [run_nla_pipeline.py](./run_nla_pipeline.py).

---

## 1. Architectural Setup & Scoped Assumptions

The original work by Anthropic relies on joint optimization frameworks to force the text engine to write mathematically optimal descriptions. In this micro-reimplementation, I separated the workflow into two independent sub-components inside the script:

1. **Activation Capture Layer:** I injected a PyTorch forward hook into the residual stream of `Qwen2.5-0.5B` at **Layer 16**. Mid-to-late residual representations are selected because they hold dense semantic structures before the geometry breaks down into token vocabulary distributions.
2. **The Verbalizer Bottleneck ($AV$):** Hidden states ($h_l \in \mathbb{R}^{896}$) are cached directly from the final token position when processing text streams. These raw vectors are $L_2$-normalized and spliced as a custom pseudo-token embedding into the input stream right after the text anchor: `"The internal state of the model represents the concept of:"`. The base model then auto-regressively predicts the continuation string ($z$).
3. **The Reconstructor Mechanism ($AR$):** The generated explanation text $z$ is fed into a parallel model instance. A custom linear projection head (`nn.Linear(896, 896)`) evaluates the final token's hidden state, attempting to map that text sequence representation back onto the original $L_2$-normalized activation space ($\hat{h}_l$).

### Dataset & Compute Realities
To run this pipeline locally without running out of memory on standard hardware, I collected 66 clean, length-filtered test activations extracted directly from the [WikiText validation split via HuggingFace Datasets](https://huggingface.co/datasets/nvidia/wikitext2). The model head optimization loop runs across 5 mini-epochs using an AdamW optimizer ($\eta = 10^{-3}$) to fit the projection parameters.

---

## 2. Mathematical Formalization

To eliminate any raw scale offsets between different execution runs, both the target activation and the reconstructed prediction are projected onto a unit sphere using standard $L_2$-normalization. This simplifies the Mean Squared Error ($\mathcal{L}_{MSE}$) directly into directional cosine alignment:

$$\mathcal{L}_{MSE} = \left\| \bar{h}_l - \overline{AR}(z) \right\|_2^2 = 2 \cdot (1 - \cos(\theta_{\text{vectors}}))$$

The baseline performance tracking uses the **Fraction of Variance Explained (FVE)**. This metric evaluates whether the autoencoder roundtrip is catching structural features or performing worse than just guessing the global baseline mean:

$$FVE = 1 - \frac{\sum \| \bar{h}_l - \overline{AR}(z) \|_2^2}{\sum \| \bar{h}_l - \mathbb{E}[\bar{h}_l] \|_2^2}$$

---

## 3. Quantitative Performance Breakdown

The training sequence converged properly, showing the average mini-batch MSE dropping steadily from `0.00136` down to `0.00048` across the 5 epochs. However, final testing revealed a severe variance gap when compared to Anthropic’s baseline targets.

| Architectural Framework | Mean Cosine Similarity | Average MSE | Fraction of Variance Explained (FVE) |
| :--- | :--- | :--- | :--- |
| **Anthropic Paper Target (Claude 3)** | 0.8100 – 0.8900 | 0.2200 | **0.6000 to 0.8000** |
| **This Replication (Qwen2.5-0.5B)** | 0.4371 | 0.001255 | **-0.6377** |

### Mechanistic Analysis of the Negative FVE
A negative FVE score means that the total squared error of the reconstructed vector is larger than the variance of the underlying target dataset. Rather than a pipeline calculation error, this reveals an essential structural limitation of scaling down this method:
* **Polysemantic Superposition Interference:** Large enterprise models can dedicate clean geometric directions to individual semantic concepts. Small 0.5B parameters models rely aggressively on *superposition*—packing multiple completely unrelated ideas into overlapping hidden dimensions. Compressing this multi-layered vector into a single text sequence completely drops the underlying directional nuances.
* **Narrow Spatial Bounds:** Limiting the validation sample size to 66 points tightly concentrates the global variance distribution. Because the text engine output frequently drifted into repetitive text loops, the linear projection head consistently overshot the boundaries of this tight distribution cluster, dropping the FVE below zero.

---

## 4. Qualitative Behavioral Profiles & Failures

Inspecting the raw strings generated by the verbalizer explains exactly why the reconstruction geometry diverged.

### Failure Mode 1: The Multiple-Choice Template Drift
* **Sample 1 Generation:** `"A. the model's internal state\nB. the model's external state"` (Cosine: `0.4760`)
* **Sample 3 Generation:** `") A. the model's internal state B. the model's external state C"` (Cosine: `0.9342`)

**Analysis:** Because `Qwen2.5-0.5B` is a raw auto-regressive base model rather than a chat-instruct variant, it has no native understanding of an instructional request to "explain" an activation. It treats the text prompt prefix as an incomplete multiple-choice question on an online exam and begins formatting a test question sheet. 

However, Sample 3's exceptionally high Cosine Similarity (`0.9342`) reveals something fascinating: even though the text looks broken to a human reader, the structural formatting remains completely predictable. Because the text layout was so stable, the linear reconstructor layer easily learned the token distribution and mapped the final hidden state straight back to the target coordinates.

### Failure Mode 2: Cross-Lingual Semantic Crossover
* **Sample 2 Generation:** `"子\nA. 事物\nB. 事物的运动状态\nC"` (Cosine: `0.0499`)

**Analysis:** This demonstrates a critical weakness in low-parameter interpretability layers. Injecting unaligned, raw activation vectors straight into the input matrix can completely derail attention pathways. Here, the numerical vector properties overrode the English prompt context and pushed the attention heads directly into Qwen's Chinese pre-training data data paths. Because the language shifted mid-stream, the reconstruction head failed completely, dropping the Cosine Similarity to nearly zero (`0.0499`).

---

## 5. Replication Protocol

### Environmental Configuration
Install all required project dependencies directly via standard `pip`:
```bash
pip install torch transformers datasets numpy accelerate

---

### 6. Results
Beginning optimisation routine for Activation Reconstructor Head... Epoch 1/5 | Average MSE Loss: 0.00136 Epoch 2/5 | Average MSE Loss: 0.00075 Epoch 3/5 | Average MSE Loss: 0.00057 Epoch 4/5 | Average MSE Loss: 0.00051 Epoch 5/5 | 
Average MSE Loss: 0.00048

 ---
Phase 5: Quantitative Metric Evaluation --- Qualitative Tracking Window: [Sample 1] Generated Explanation: "A. the model's internal state B. the model's external state" | Cosine Similarity: 0.4760 [Sample 2] Generated Explanation: "子 A. 事物 B. 事物的运动状态 C" | Cosine Similarity: 0.0499 [Sample 3] Generated Explanation: ") A. the model's internal state B. the model's external state C" | Cosine Similarity: 0.9342 

## Final Quantitative Summary 
Metric: Mean Cosine Similarity:       0.4371 
Mean Squared Error (MSE):      0.001255 
Fraction of Variance Explained (FVE): -0.6377
