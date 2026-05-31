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
| Base models | DistilBERT (general), FinBERT (finance, **FPB-contaminated**), FinBERT-Tone (finance, **leakage-safe**) | Ladder separates domain pretrain vs task adapt |
| Adaptation | LoRA / DoRA / PiSSA on attention | &lt;2% trainable params; base frozen |
| Metric | **F1-macro** (+ 95% bootstrap CI, reported test N) | Class imbalance (~60% neutral); small test set |
| Evidence | Held-out test + validation checkpointing | Required for internal review |

**Reading the headline number correctly.** The often-quoted "test F1-macro ~0.31 → ~0.91" (DistilBERT, 3 epochs) is *not* a measure of what LoRA buys. The 0.31 is a **random, untrained classification head** on a frozen encoder — a chance-level floor that measures "did we train anything at all," so the ~0.60 jump is dominated by training a head, not by LoRA. The honest, load-bearing comparison is **`head_only` vs `lora` at the same learning-rate budget and epoch count**: this isolates the marginal value of adapting attention over a trained linear head, and in the smoke run it is only a few points (~0.48 → ~0.51 at 1 epoch). The baseline ladder prints this `head_only`-vs-`lora` delta explicitly and labels every `eval_only` row by `baseline_type` (`random_floor` / `zero_shot` / `leaked_anchor`). Do not cite `eval_only → lora` as the LoRA lift.

---

## 5. Success metrics

### Phase 1 — Proof of mechanism (current repo)

| Metric | Target | Status |
|--------|--------|--------|
| Baseline test F1-macro documented (with test N) | Required | Main notebook + ladder CSV |
| **`lora` − `head_only`** F1 lift (the real LoRA contribution), same LR budget | Demonstrate a CI-separated lift | Ladder prints per-backbone delta; treat ~0.60 random-head jump as *not* the LoRA effect |
| Per-run **95% bootstrap CI** + multi-seed mean ± std | No claim below CI/seed-std resolution | `evaluate_f1_macro(compute_ci=True)`; seeds 42/43/44 |
| Trainable parameters | **&lt; 5%** of total | ~1.1% (LoRA + head) |
| Adapter reload inference | Matches in-session eval | Main notebook §8 |
| Leakage-safe domain baseline (FinBERT-Tone, not FPB-trained) | Replace contaminated FinBERT anchor | `finbert_tone_*` ladder rows |
| Forgetting probe (general sentiment after finance adapter) | Δ on SST-2 reported | `experiments/forgetting_probe.py` |

### Phase 2 — Organizational readiness (next)

| Metric | Target |
|--------|--------|
| Slice eval | F1 on negation, length bins, entity-heavy sentences |
| Calibration | ECE / reliability on confidence scores |
| LR sweep before exotic PEFT | Tune LoRA LR first; DoRA/PiSSA often match vanilla LoRA once LR is tuned |

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
| `experiments/baseline_ladder.ipynb` | Compare eval-only, head-only, LoRA, DoRA, PiSSA × DistilBERT/FinBERT/FinBERT-Tone |
| `experiments/run_baseline_ladder.py` | Headless ladder run → `experiments/results/*.csv` (prints `head_only`-vs-`lora` key comparison + per-run CIs) |
| `experiments/forgetting_probe.py` | Evaluate a saved finance adapter on general sentiment (SST-2) to probe catastrophic forgetting |
| `finetuning_lib/` | Shared data, training, evaluation code |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Quick smoke (1 epoch, seed 42)
python experiments/run_baseline_ladder.py --quick

# Full ladder (3 epochs, seeds 42/43/44) — longer
python experiments/run_baseline_ladder.py

# Forgetting probe on a saved adapter
python experiments/forgetting_probe.py \
    --checkpoint experiments/results/checkpoints/distilbert_lora_seed42 \
    --backbone distilbert
```

---

## 8. Governance and risk notes

- **FinBERT (`ProsusAI/finbert`) is contaminated for this benchmark.** It was fine-tuned *on FinancialPhraseBank*, so its `eval_only` score (~0.97, ≈ the 97% test accuracy reported in the original FinBERT blog on the all-agree subset) is an **in-sample / training-set-overlap** number, **not** a zero-shot transfer baseline. Treat it as a *leaked upper anchor* only. Crucially, **do not** build the flagship "domain encoder + LoRA" experiment on FinBERT and then evaluate on FPB — any measured lift would be on data the backbone has already seen. The ladder flags these rows with `contaminated=True` and `baseline_type="leaked_anchor"`.
- **Leakage-safe domain baseline.** For a fair finance-encoder comparison the ladder adds **FinBERT-Tone (`yiyanghkust/finbert-tone`)**, trained on analyst-report / forward-looking-statement sentences rather than FPB. Its `eval_only` is a genuine zero-shot baseline (`baseline_type="zero_shot"`). Provenance should still be re-verified before any external claim.
- **Statistical resolution.** The all-agree test split is ~340 sentences; the 95% bootstrap CI on macro-F1 is ≈ ±0.03–0.05. Differences smaller than the CI width (or smaller than the across-seed std) are **not** results. Every run now reports a per-run CI and test N, and the full ladder reports mean ± std over seeds 42/43/44. Report N alongside every F1.
- **`eval_only` is not a LoRA baseline.** On DistilBERT, `eval_only` uses a *random untrained head* (`baseline_type="random_floor"`, chance level). The honest LoRA contribution is `lora − head_only` at the same LR budget (locked via `LOCKED_LEARNING_RATE`), which the runner prints explicitly.
- **Tune LR before reaching for DoRA/PiSSA.** Recent work indicates that once the LoRA learning rate is properly tuned, vanilla LoRA, DoRA, and PiSSA reach similar peak performance (within ~1–2%). Run an LR sweep before attributing gains to an exotic PEFT variant.
- **Internal data** must not be committed; train only on approved stores.  
- **Class imbalance** makes accuracy misleading; F1-macro is necessary but not sufficient for asymmetric error costs.  
- **“Zero-shot”** in the main notebook means an **untrained classification head** on a general encoder—not a finance instruction-tuned LLM.

---

## 9. References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022  
- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, ICML 2024  
- Meng et al., *PiSSA*, NeurIPS 2024  
- Malo et al., FinancialPhraseBank, JASIST 2014  
- Araci, *FinBERT* (`ProsusAI/finbert`), 2019 — **fine-tuned on FinancialPhraseBank**; ~97% all-agree test accuracy (treated here as a leaked anchor)  
- Huang et al., *FinBERT: A Large Language Model for Extracting Information from Financial Text* (`yiyanghkust/finbert-tone`), Contemporary Accounting Research 2023 — analyst-report tone model used as the leakage-safe finance baseline  
- Socher et al., *Recursive Deep Models for Semantic Compositionality* (SST / SST-2), EMNLP 2013 — general-domain sentiment used for the forgetting probe  
- *Learning Rate Matters: Vanilla LoRA May Suffice* (2026) — tuned-LR LoRA matches DoRA/PiSSA within ~1–2% *(verify exact arXiv ID before external citation)*  

> Citation hygiene: every reference above should be opened and confirmed (ID, venue, and the specific claim it supports) before being repeated in any external or executive-facing document. Do not cite a bare `arxiv.org` link.

---

*Last updated: corrected baseline-ladder framing — `head_only`-vs-`lora` as the LoRA contribution, FinBERT contamination flagged, FinBERT-Tone leakage-safe baseline, per-run bootstrap CIs + reported test N, and a SST-2 forgetting probe.*
