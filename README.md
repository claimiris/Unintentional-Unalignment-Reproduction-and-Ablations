# Unintentional Unalignment: Reproduction & Ablation Study

This repository contains the reproduction of the paper **"Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization"** and subsequent ablation studies to test the robustness of the findings.

The project validates how Direct Preference Optimization (DPO) can unintentionally decrease the likelihood of preferred responses and shift probability mass to harmful or opposite responses, a phenomenon termed Likelihood Displacement.

## Original Paper

**Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization** *Noam Razin, Sadhika Malladi, Adithya Bhaskar, Danqi Chen, Sanjeev Arora, Boris Hanin* (2024)

> **Abstract:** The paper demonstrates that likelihood displacement can be catastrophic, shifting probability mass from preferred responses to responses with an opposite meaning. For example, training a model to prefer "No" over "Never" can sharply increase the probability of "Yes".

* **arXiv Link:** [https://arxiv.org/abs/2410.08847](https://arxiv.org/abs/2410.08847)
* **Official Code:** [https://github.com/princeton-nlp/unintentional-unalignment](https://github.com/princeton-nlp/unintentional-unalignment)

## Reproduced Results

Successfully replicated the three major findings of the original paper:

1.  **Catastrophic Likelihood Displacement (Section 3):**
    * Confirmed that when optimizing for specific preferences (e.g., "No" > "Never"), the probability of the preferred token ("No") often decreases.
    * Verified that probability mass shifts to opposite tokens (e.g., "Yes"), leading to "catastrophic" displacement.
    * **Models:** OLMo-1B and Gemma-2-9B (reproduced with T4 and H100 GPUs).

2.  **Identifying Sources with CHES (Section 5):**
    * Validated that the **Centered Hidden Embedding Similarity (CHES)** score is a more effective metric than edit distance for predicting likelihood displacement.
    * Plots confirmed that high-CHES samples contribute most significantly to the probability drops.

3.  **Safety Alignment Efficacy (Section 6):**
    * Reproduced the drop in refusal rates on safety benchmarks (e.g., "SorryBench") after DPO training.
    * Confirmed that filtering training data based on CHES scores helps maintain higher refusal rates compared to standard DPO.

*(The `End Eval Report.pdf` in this repository contains the detailed quantitative breakdown.)*

## Ablation Studies

Additional ablations to investigate the internal mechanics and mitigation strategies.

### Ablation 1: Layer-wise Metric Analysis
Analyzed similarity metrics (CHES, Inner Product) across all 16 layers of the OLMo-1B model to understand *where* the displacement originates. Compared Pre-Norm and Post-Norm values to isolate the effects of Attention vs. MLP blocks.
Performed similar studies on Gemma-2b-it model with the SORRYBench dataset to analyse the roles of different layer regions in generating the final response.

### Ablation 2: Soft-Filtering
Instead of hard-removing samples (as done in the original paper), I have investigated "soft-filtering" by downweighting samples based on their length normalised CHES score. This allows the model to still learn from the data without suffering catastrophic displacement.

### Ablation 3: Flipped Model (The "Reverse" Experiment)
Investigated the effects of inverting preference pairs to observe if probability shifts occur in the counter-intuitive opposite direction.

* **Download Link:** [Google Drive - Ablation 3 Flipped Model](https://drive.google.com/drive/folders/1mc4mFcIbDMucHRPACTMFhyzmRCn6UGNh?usp=sharing)
