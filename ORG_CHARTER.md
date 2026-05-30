# Organizational Charter: Domain-Adaptive Financial Sentiment

This document defines **why** the finetuning implementation exists, **what question** it answers for a large organization, and **how success is measured**. It complements the technical notebooks (`McCoy_LoRA_Finetuning_Framework.ipynb`, `experiments/baseline_ladder.ipynb`).

---

## 1. Business context

Large enterprises in **financial services, insurance, and regulated industries** deploy language models on text that differs materially from general web corpora: investor-oriented sentiment, filings, policy language, and internal memos. General-purpose models and ad hoc prompting introduce:

- **Inconsistent labels** across teams and channels  
- **High marginal cost** at batch volume (API-per-document)  
- **Weak auditability** (prompt changes are not versioned like model artifacts)  
- **Data residency constraints** (sensitive text must stay in-house)

This repository demonstrates **parameter-efficient adaptation (LoRA/PEFT)**: frozen foundation weights plus small, swappable adapters—an operational pattern that scales to many tasks without full model retraining.

---

## 2. Key question (executive framing)

> **Can we obtain reliable, auditable, domain-specific classification behavior by adapting a shared foundation model with a small fraction of trainable parameters—such that measured quality and total cost of ownership beat generic APIs and untuned baselines on our workloads?**

### Sub-questions (model risk, engineering, product)

| # | Question | Owner |
|---|----------|--------|
| Q1 | **Efficacy** — How much lift vs untuned baseline and vs finance-pretrained models (e.g. FinBERT)? | ML / Research |
| Q2 | **Efficiency** — Cost and calendar time to retrain when labels or policy change? | ML Platform |
| Q3 | **Safety of adaptation** — Does domain tuning degrade non-finance behavior (forgetting proxy)? | Model Risk |
| Q4 | **Operability** — Can we version, roll back, and promote adapters independently of the base model? | MLOps |
| Q5 | **Sustainability** — Does the approach survive base-model upgrades and new tasks? | Architecture |

The main notebook answers **Q1–Q2** on a public benchmark. The baseline ladder (`experiments/`) answers **Q1** in more depth (method and backbone comparisons). **Q3–Q5** require additional experiments and platform work (see §6).

---

## 3. Scope: what we are solving

| In scope | Out of scope (different tools) |
|----------|--------------------------------|
| Fixed-label **sentiment** (negative / neutral / positive) | Open-ended generation quality |
| **Short financial news sentences** (FinancialPhraseBank) | Long-document QA over filings (needs long-context + RAG) |
| **Weight-based adaptation** with frozen base | Dynamic factual knowledge (RAG, knowledge graphs) |
| **Adapter artifacts** (~MB) for promotion/rollback | Full-model fork per task |
| Reproducible **baseline vs adapted** metrics | Production API, auth, monitoring (future phase) |

---

## 4. Technical approach (summary)

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Task | 3-way investor sentiment | Standard, auditable taxonomy |
| Data | FinancialPhraseBank (all-agree subset) | Expert labels; public benchmark |
| Base models | DistilBERT (general), FinBERT (finance) | Ladder separates domain pretrain vs task adapt |
| Adaptation | LoRA / DoRA / PiSSA on attention | &lt;2% trainable params; base frozen |
| Metric | **F1-macro** | Class imbalance (~60% neutral) |
| Evidence | Held-out test + validation checkpointing | Required for internal review |

Reference result (main notebook, DistilBERT + LoRA, 3 epochs): test F1-macro **~0.31 → ~0.91** vs untrained head on the same split.

---

## 5. Success metrics

### Phase 1 — Proof of mechanism (current repo)

| Metric | Target | Status |
|--------|--------|--------|
| Baseline test F1-macro documented | Required | Main notebook |
| Adapted test F1-macro &gt; baseline + **0.15** absolute | Demonstrate clear lift | Achieved (~+0.60) |
| Trainable parameters | **&lt; 5%** of total | ~1.1% (LoRA + head) |
| Adapter reload inference | Matches in-session eval | Main notebook §8 |
| Baseline ladder CSV | All methods ranked on same split | `experiments/run_baseline_ladder.py` |

### Phase 2 — Organizational readiness (next)

| Metric | Target |
|--------|--------|
| Multi-seed stability | Mean ± std F1 over ≥3 seeds |
| FinBERT off-the-shelf vs LoRA-on-FinBERT | Quantify marginal value of task LoRA |
| Slice eval | F1 on negation, length bins, entity-heavy sentences |
| Forgetting proxy | Eval on non-finance sentiment after finance adapter |
| Calibration | ECE / reliability on confidence scores |

### Phase 3 — Production (future)

| Metric | Target |
|--------|--------|
| Promotion gate | No deploy if test F1 &lt; threshold vs golden set |
| Latency p95 | Within SLO for batch + online paths |
| Retrain cadence | Documented $/run and time-to-promote |
| Registry | Adapter version, data hash, config hash, approver |

---

## 6. Decision matrix: when to use this approach

| Scenario | Recommendation |
|----------|----------------|
| Stable task, hundreds–thousands of labels, many task variants | **LoRA/PEFT** (this repo) |
| Facts change daily (rates, policy) | **RAG**, not finetuning alone |
| &lt;100 labels, exploratory | Prompting / few-shot first |
| Need open-ended reasoning or tool use | SFT + RL / agent stack (e.g. FINDAP-style pipelines) |
| Full behavior rewrite, large budget | Full SFT (higher forgetting risk) |

---

## 7. Artifacts and how to run

| Artifact | Purpose |
|----------|---------|
| `McCoy_LoRA_Finetuning_Framework.ipynb` | End-to-end LoRA tutorial + adapter save/load |
| `experiments/baseline_ladder.ipynb` | Compare eval-only, head-only, LoRA, DoRA, PiSSA × DistilBERT/FinBERT |
| `experiments/run_baseline_ladder.py` | Headless ladder run → `experiments/results/*.csv` |
| `finetuning_lib/` | Shared data, training, evaluation code |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Quick smoke (1 epoch, seed 42)
python experiments/run_baseline_ladder.py --quick

# Full ladder (3 epochs, seeds 42/43/44) — longer
python experiments/run_baseline_ladder.py
```

---

## 8. Governance and risk notes

- **FinBERT eval-only** uses publicly pretrained weights; treat as an external dependency with its own license and lineage. FinBERT uses a different `label2id` order than the PhraseBank export—the ladder remaps labels automatically (`finetuning_lib.data.remap_dataset_labels`).  
- **Internal data** must not be committed; train only on approved stores.  
- **Class imbalance** makes accuracy misleading; F1-macro is necessary but not sufficient for asymmetric error costs.  
- **“Zero-shot”** in the main notebook means an **untrained classification head** on a general encoder—not a finance instruction-tuned LLM.

---

## 9. References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022  
- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, ICML 2024  
- Meng et al., *PiSSA*, NeurIPS 2024  
- Malo et al., FinancialPhraseBank, JASIST 2014  
- Yang et al., *FinDAP* / domain-adaptive post-training for financial LLMs, EMNLP 2025  
- Araci, *FinBERT*, 2019  

---

*Last updated: aligned with repository implementation including baseline ladder and shared `finetuning_lib`.*
