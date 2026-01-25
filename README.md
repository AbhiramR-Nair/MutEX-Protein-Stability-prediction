# MutEX: Physics-Aware Protein Stability Prediction

## Overview

MutEX is a machine learning framework for predicting protein stability changes (ΔΔG) caused by **single amino acid mutations**. The core goal of this project is not only to achieve high predictive accuracy, but to ensure that predictions remain **thermodynamically and physically consistent**, specifically by enforcing the **antisymmetry property of ΔΔG**.

This project addresses a key limitation in many existing protein stability predictors: models may fit experimental data well, yet violate fundamental physical laws, making them unreliable for real biological or engineering use.

---

## Literature Review
Recent advances in protein stability prediction have been driven by large-scale experimental datasets and protein language models (PLMs). Early approaches relied on handcrafted structural features and physics-based energy functions, which struggled to generalize across proteins and mutation contexts.

With the emergence of PLMs trained on millions of sequences, several studies have demonstrated that learned sequence embeddings implicitly encode structural and thermodynamic information. Transfer learning from these models has significantly improved ΔΔG prediction performance, even with relatively simple downstream networks.

However, a key limitation in much of the existing literature is the lack of explicit enforcement of physical constraints, particularly antisymmetry of ΔΔG. Many models achieve strong correlation or low RMSE while producing predictions that violate basic thermodynamic principles, limiting their scientific reliability.

Recent work has begun exploring larger datasets and hybrid architectures, but physics-aware learning objectives remain underexplored. MutEX directly addresses this gap by integrating antisymmetry into the training objective itself, rather than treating it as a post hoc evaluation criterion.

## Motivation

Protein stability determines whether a protein folds correctly and functions as intended. Even a single-point mutation can:

* Destabilize folding
* Reduce biological activity
* Cause disease or drug resistance

While experimental measurement of mutation effects is the gold standard, it is:

* **Slow** (weeks to months per mutation)
* **Expensive**
* **Not scalable** to millions of possible mutations

MutEX aims to solve this by using pretrained **Protein Language Models (PLMs)** and physics-aware training to predict ΔΔG at scale.

---

## What is ΔΔG and Why Antisymmetry Matters

* **ΔG** measures protein folding stability
* **ΔΔG** measures the change in stability caused by a mutation

Interpretation:

* ΔΔG < 0 → stabilizing mutation
* ΔΔG > 0 → destabilizing mutation

### Antisymmetry Principle

If a mutation A → B has ΔΔG = +x, then the reverse mutation B → A must have ΔΔG = −x.

Many ML models violate this constraint, producing physically inconsistent predictions. **MutEX explicitly enforces antisymmetry during training**, ensuring thermodynamic validity by design.

---

## Dataset

**Source:** K50 dataset (Tsuboyama et al., 2022, Zenodo)

**Raw scale:** 851,552 experimentally measured mutations

### Cleaning and Validation

The raw dataset contained:

* Wild-type (WT) entries
* Multiple mutations per sample
* Insertions and deletions
* Inconsistent mutation annotations

**Cleaning steps included:**

* Removing WT entries
* Retaining only single-point substitutions
* Excluding insertions/deletions
* Standardizing mutation notation using regex
* Verifying WT and mutant residue correctness

**Final dataset:**

* 375,560 validated single-point mutations
* 11 curated columns
* Used for embedding extraction and model training

---

## Exploratory Data Analysis (EDA)

EDA was performed to assess data quality, bias, and modeling risks:

* No missing values or duplicate samples
* ΔΔG distribution centered near 0 with long tails
* Destabilizing mutations dominate (~79%), reflecting biological reality
* Sequence length shows negligible correlation with ΔΔG (Pearson ≈ 0.05)
* Strong **cluster-level imbalance** across proteins

### Critical Insight: Data Leakage Risk

Random train–test splits lead to homologous proteins appearing in both sets, artificially inflating performance. To prevent this, **cluster-level splitting** was enforced.

---

## Protein Language Models (PLMs)

MutEX uses pretrained **ESM2** models to generate contextual embeddings for both wild-type and mutant sequences. These embeddings capture:

* Local residue environment
* Long-range interactions
* Evolutionary constraints

PLMs enable learning stability-relevant features directly from sequence data without hand-crafted descriptors.

---

## Model Architecture

### Two-Phase Training Pipeline

**Phase 1 – Embedding Extraction**

* Generate embeddings for WT and mutant sequences using pretrained ESM2

**Phase 2 – Prediction Network**

* Lightweight attention + fully connected layers
* Regression target: ΔΔG

---

## Model Variants

### Model Version 1 (Baseline)

* Uses WT and mutant embeddings directly
* Antisymmetry evaluated post-training
* High accuracy but physically inconsistent

### Model Version 2 (Physics-Aware)

* Uses averaged embeddings from last 4 PLM layers
* Derived mutation features:

  * Δ embedding
  * |Δ| embedding
  * Cosine similarity
  * L2 distance
* **Antisymmetry enforced during training via regularization loss**

---

## Results Summary

* Model-1 variants achieve strong Pearson/Spearman correlation but violate antisymmetry
* Model-2 (MV_2.1.1) achieves:

  * High antisymmetry correlation (~0.86)
  * Strong reduction in prediction bias
  * Competitive RMSE and MAE

Although Model-2 sacrifices a small amount of raw error performance, it is the **only model that produces physically meaningful ΔΔG predictions**.

---

## Deployment Architecture

* **Client (CPU):** Input parsing and validation via Streamlit
* **Server (GPU):** ESM2-650M embedding extraction via REST API
* **Client (CPU):** Feature construction and ΔΔG prediction
* **Physics check:** Forward and reverse mutation predictions to verify antisymmetry
* **Output:** Predictions displayed and downloadable as CSV

---

## Team Contributions

* **Data Collection, Cleaning, and EDA:** Teammate

  * Dataset acquisition and validation
  * Cleaning and preprocessing
  * Exploratory data analysis and bias assessment

* **Embedding Extraction and Model Training:** Abhiram

  * PLM embedding pipeline
  * Feature engineering
  * Model architecture design
  * Physics-aware training and evaluation

---

## Future Scope

* Extend to longer and multi-domain proteins
* Support multi-point mutation prediction
* Integrate additional thermodynamic constraints

---

## References
* Pak, M., Dovidchenko, N., Sharma, S., & Ivankov, D. (2023).
The new mega dataset combined with a deep neural network makes progress in predicting the impact of single mutations on protein stability.
bioRxiv. https://doi.org/10.1101/2022.12.31.522396
* Tsuboyama, K., et al. (2022).
Mega-scale experimental analysis of protein folding stability.
Zenodo Dataset.
* H. Dieckhaus,M. Brocidiacono,N.Z. Randolph, & B. Kuhlman,  Transfer learning to leverage larger datasets for improved prediction of protein stability changes,
Proc. Natl. Acad. Sci. U.S.A. 121 (6) e2314853121, https://doi.org/10.1073/pnas.2314853121 (2024).

---

## Disclaimer

This project is intended for research and educational purposes. Predictions should not be used directly for clinical or therapeutic decision-making without experimental validation.
