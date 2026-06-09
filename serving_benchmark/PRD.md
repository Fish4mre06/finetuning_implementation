# PRD — LoRA Inference Serving Benchmark: vLLM vs TensorRT-LLM vs NVIDIA NIM (single L4)

**Status:** Draft v1 · **Owner:** james@krazyape.com · **Last updated:** 2026-06-09
**Related:** [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md) · repo [`../ORG_CHARTER.md`](../ORG_CHARTER.md)

---

## 1. One-liner

Take a fine-tuned **finance LoRA adapter on Llama-3.1-8B-Instruct**, serve the same
artifact through a **naive baseline**, then through **vLLM**, **TensorRT-LLM**, and
**NVIDIA NIM** on a **single L4 24 GB GPU**, and produce a reproducible
**before/after latency + throughput benchmark** with a written analysis. The output
is portfolio-grade Solutions-Architect collateral: "here's a benchmark I ran," not
"I'd ramp on the stack."

---

## 2. Background & motivation

The NVIDIA-aligned SA role is hired to produce **reusable, defensible benchmarks**
on the inference stack. The existing repo demonstrates *training-side* PEFT
(encoder classification with LoRA/DoRA/PiSSA). This project extends the same
"adapters as cheap, swappable artifacts" thesis to the **serving side**, where the
questions that decide a deployment are latency, throughput, cost-per-token, and how
cleanly you can host *many* LoRA adapters behind *one* base model.

We deliberately target a **single L4** rather than an A100/H100. The L4 is the
realistic cost-efficient inference tier, and its constraints make the engineering
story sharper:

| L4 property | Consequence for this benchmark |
|---|---|
| 24 GB GDDR6 | Llama-3.1-8B in **fp16 ≈ 16 GB weights** leaves only ~6 GB for KV cache + activations → concurrency is memory-bound. **Quantization is not optional, it's the headline.** |
| ~300 GB/s memory bandwidth | Decode is **bandwidth-bound**; tokens/sec is gated by memory traffic, so KV-cache precision and paging matter as much as compute. |
| Ada Lovelace, 4th-gen Tensor Cores | Native **FP8** support → TensorRT-LLM FP8 / AWQ-int4 are the levers that unlock batch size and throughput. |
| 72 W single-slot | The "perf-per-watt / perf-per-dollar" narrative an SA actually sells. |

So the benchmark is not just "which server is fastest" — it's "**on a constrained,
realistic GPU, what does each stack let you do with one fine-tuned model, and what
does quantization buy you?**"

---

## 3. Goals / Non-goals

### Goals
- **G1.** One fine-tuned finance LoRA adapter, served unchanged across all stacks.
- **G2.** A **before/after** comparison: naive baseline → optimized stacks (vLLM, TRT-LLM, NIM).
- **G3.** Standard inference metrics — **TTFT, ITL/TPOT, end-to-end latency (p50/p90/p99), output-token throughput, request throughput** — across a **concurrency sweep**.
- **G4.** A **precision axis** (fp16 → FP8 → AWQ-int4) showing how quantization unlocks concurrency on a 24 GB card.
- **G5.** A **multi-LoRA axis**: serve N adapters behind one base and measure the per-adapter overhead vs a single merged model.
- **G6.** A **correctness gate** — verify each deployment produces equivalent outputs *before* any performance number is trusted.
- **G7.** Fully **reproducible**: pinned containers, one-command runs, results + plots committed.

### Non-goals
- Multi-GPU / tensor-parallel scaling (single L4 only).
- Training a production-quality finance model (the adapter need only be a *real*, non-trivial LoRA — quality is a correctness gate, not the deliverable).
- Autoscaling, Kubernetes, or a production gateway (future phase; noted in §11).
- Beating published vendor numbers — the deliverable is a **fair, documented, reproducible** comparison, not a leaderboard score.

---

## 4. Success metrics (acceptance criteria)

| # | Criterion | Target |
|---|---|---|
| A1 | Same adapter served on **≥4 configurations** (baseline + vLLM + TRT-LLM + NIM) | Required |
| A2 | Metrics captured: TTFT, ITL, e2e p50/p90/p99, output tok/s, req/s, GPU util, peak VRAM | All, per config |
| A3 | Concurrency sweep | ≥ {1, 4, 8, 16, 32, 64} (until OOM / saturation) |
| A4 | **Correctness parity** vs baseline before perf is reported | ≥ 0.99 greedy-decode token agreement OR task-metric within CI |
| A5 | Each perf number is a **median of ≥3 runs** with dispersion reported (IQR or 95% CI) | Required — single-run numbers are not results |
| A6 | Quantization study (fp16 vs FP8 vs AWQ-int4) on max sustainable concurrency + throughput | Required |
| A7 | Multi-LoRA overhead (1 vs 4 vs 16 adapters) quantified | Required |
| A8 | One-command repro per stack + committed `results/*.csv` + plots + written analysis | Required |

> **Anti-goal restated as a gate (A4/A5):** a fast server emitting wrong tokens is a
> bug, not a win. We learned this on the training side (a leaked/contaminated
> baseline produces an impressive-but-meaningless number); the serving analogue is
> benchmarking a mis-loaded adapter or a broken quantization. **Correctness first,
> then speed.**

---

## 5. Users / personas (who consumes this collateral)

| Persona | What they take from it |
|---|---|
| **NVIDIA SA / hiring panel** | Evidence the candidate can stand up TRT-LLM + NIM + vLLM, run a fair benchmark, and explain the trade-offs on real hardware. |
| **Enterprise ML platform lead** | A template to evaluate serving stacks for their own LoRA fleet on cost-efficient GPUs. |
| **Applied ML engineer** | A working harness + commands to reproduce and extend to their model. |

---

## 6. Scope: model, adapter, hardware

| Dimension | Choice | Rationale |
|---|---|---|
| Base model | **meta-llama/Llama-3.1-8B-Instruct** | Ubiquitous, well-supported by all three stacks; 8B fits a single L4 with quantization. |
| Adapter | **Finance LoRA** trained on a generative reframe of the repo's sentiment / finance-QA data | Keeps the repo's domain thesis; a real adapter (rank 16–32) exercises LoRA paths. |
| Task framing | Instruction → finance sentiment label / short rationale | Deterministic enough for a greedy-decode correctness gate. |
| GPU | **1× NVIDIA L4 24 GB** (Ada) | Cost-efficient inference tier; quantization-forced; FP8-capable. |
| Precisions | fp16 (ref), **FP8** (TRT-LLM/NIM), **AWQ-int4** (vLLM/TRT-LLM) | The lever that makes 8B practical at batch on 24 GB. |

---

## 7. Functional requirements

- **FR1.** Train/produce one finance LoRA adapter (rank, alpha, target modules documented) and export in a portable format (PEFT/HF) consumable by all stacks.
- **FR2.** Provide a **baseline server** (HF Transformers `generate`, single-stream, no continuous batching) as the "before."
- **FR3.** Stand up **vLLM** with `--enable-lora` (single and multi-LoRA) and AWQ option.
- **FR4.** Build a **TensorRT-LLM** engine with LoRA (`trtllm-build`, LoRA plugin) in fp16 and FP8 (ModelOpt quantization).
- **FR5.** Deploy **NVIDIA NIM** for Llama-3.1-8B with the LoRA adapter mounted (dynamic/multi-LoRA), OpenAI-compatible endpoint.
- **FR6.** A **single load-generation harness** that hits every stack's OpenAI-compatible endpoint identically (same prompts, sampling params, input/output lengths), using **NVIDIA GenAI-Perf** as the primary tool (vLLM's `benchmark_serving.py` as a cross-check).
- **FR7.** A **correctness harness** that compares each deployment's greedy outputs against the baseline on a fixed prompt set.
- **FR8.** A **results layer**: normalized CSV schema, aggregation across repeat runs (median + dispersion), and plot generation (latency-vs-throughput "knee" curves, VRAM-vs-concurrency, multi-LoRA overhead bars).

---

## 8. Methodology

### 8.1 Fairness controls (the credibility of the whole exercise)
- **Identical workload** across stacks: same prompt dataset, same input/output token profiles, same sampling params, same tokenizer.
- **Warmup** runs discarded; steady-state measured.
- **Same decode settings** for the correctness gate (greedy / temperature 0).
- **Isolated GPU** (no co-tenants), pinned driver/CUDA, fixed clocks where possible.
- **Pinned container digests** per stack; versions recorded in `run_config.json`.
- Where a knob has no equivalent across stacks (e.g. `max_num_seqs` vs engine `max_batch_size`), **document the mapping and report both**, rather than silently "tuning to win."

### 8.2 Workload profiles
| Profile | Input tok | Output tok | Represents |
|---|---|---|---|
| **Chat-short** | 256 | 128 | Interactive classification / short rationale |
| **RAG-long** | 2048 | 256 | Document-grounded finance Q&A (stresses KV cache on L4) |
| **Batch-skew** | 512 | 512 | Throughput-oriented offline scoring |

### 8.3 Metrics
- **Latency:** TTFT (time-to-first-token), ITL/TPOT (inter-token latency), end-to-end **p50/p90/p99**.
- **Throughput:** output tokens/sec (system), requests/sec (goodput at a latency SLO, e.g. p90 TTFT < 500 ms).
- **Efficiency:** peak VRAM, GPU utilization, **tokens/sec/GB**, est. **$/1M tokens** (L4 on-demand rate, documented).
- Reported as **latency–throughput curves** (sweep concurrency) — the "knee" is the deliverable, not a single point.

### 8.4 Experimental matrix
```
stacks   = {baseline-HF, vLLM, TRT-LLM, NIM}
precision= {fp16, FP8, AWQ-int4}        # not all valid on all stacks; matrix notes gaps
workload = {chat-short, RAG-long, batch-skew}
concurr. = {1, 4, 8, 16, 32, 64, ...}   # until OOM/saturation
adapters = {0 (base), 1 (single LoRA), 4, 16 (multi-LoRA)}
repeats  = 3   # median + dispersion
```
Not every cell runs — the plan (see Implementation Plan §4) defines the **priority slice**: baseline vs vLLM vs TRT-LLM vs NIM at fp16 + the precision study + the multi-LoRA study, on `chat-short` first, then `RAG-long`.

### 8.5 Statistical rigor (carried over from the repo's training work)
Every reported number is the **median of ≥3 runs with dispersion** (IQR or bootstrap CI). Differences smaller than run-to-run dispersion are **not** claimed as wins. This mirrors the bootstrap-CI / reported-N discipline already adopted in `finetuning_lib`.

---

## 9. Stack-specific requirements & expected story

> Numbers below are **directional hypotheses to be measured**, not results. They set
> expectations and define what "success" looks like; the benchmark replaces them
> with measured values + dispersion.

| Stack | Role | LoRA support | Precision on L4 | Expected story |
|---|---|---|---|---|
| **Baseline (HF `generate`)** | "Before" | Merge or attach PEFT | fp16 | Single-stream, no continuous batching → low throughput, this is the bar to beat. |
| **vLLM** | Optimized OSS | `--enable-lora`, multi-LoRA, `--max-loras` | fp16, AWQ-int4 | PagedAttention + continuous batching → large throughput jump; multi-LoRA nearly free; easiest to stand up. |
| **TensorRT-LLM** | Optimized NVIDIA (hand-built) | LoRA plugin, `trtllm-build` | fp16, **FP8** (ModelOpt) | Best per-GPU latency/throughput after engine build + FP8; highest setup cost; the "max performance" anchor. |
| **NVIDIA NIM** | Optimized NVIDIA (turnkey) | dynamic/multi-LoRA, OpenAI API | auto-selected optimized profile (often TRT-LLM FP8 under the hood) | Closest to enterprise reality: container + adapter mount, minimal tuning. Note: since NIM may use a TRT-LLM backend, frame the NIM-vs-TRT-LLM comparison as **operational ergonomics + auto-profile vs hand-built engine**, not two unrelated engines. |

**L4 memory budget (Llama-3.1-8B, illustrative, to be confirmed empirically):**

| Precision | ~Weights | KV headroom in 24 GB | Implication |
|---|---|---|---|
| fp16 | ~16 GB | ~6 GB | Low max concurrency; OOM at long context. |
| FP8 | ~8 GB | ~14 GB | ~2× KV headroom → materially higher batch & throughput. |
| AWQ-int4 | ~5.5 GB | ~16 GB | Highest concurrency; watch for quality/correctness regression at the gate. |

---

## 10. Deliverables / artifacts

1. **`serving_benchmark/`** scaffold: per-stack launch scripts, the load + correctness harness, results schema, plotting.
2. **`results/*.csv`** — normalized metrics across the matrix, repeat-aggregated.
3. **Plots** — latency-vs-throughput knee curves per stack; VRAM-vs-concurrency; precision study; multi-LoRA overhead.
4. **`REPORT.md`** — the written analysis an SA hands over: setup, fairness controls, findings, recommendation per scenario, and honest caveats.
5. **One-command repro** per stack (Make targets / scripts) + pinned container digests.
6. **Correctness report** establishing output parity before any perf claim.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Gated model access** (Llama-3.1 license) | Document HF/NGC access steps; provide a non-gated fallback base (e.g. Mistral-7B / a permissive 8B) so the harness runs without the gate. |
| **L4 OOM at fp16 + long context** | Treat as a *finding*, not a failure; it motivates the quantization axis. Cap concurrency sweep at OOM and record it. |
| **Quantization changes outputs** | The §4 A4 correctness gate runs per precision; report any quality delta alongside the speedup — never a speedup in isolation. |
| **NIM ≈ TRT-LLM backend overlap** | Explicitly frame NIM as the *turnkey/ergonomics* comparison; report the auto-selected profile so the overlap is transparent. |
| **Unfair tuning** | Document every knob; where no cross-stack equivalent exists, report both settings. No silent tuning-to-win. |
| **No GPU in this dev container** | PRD + plan + harness are authored here; **actual runs require an L4 host** (cloud instance or NVIDIA AI workstation). The plan flags every GPU-dependent step. |
| **Vendor version drift** | Pin container digests + record versions in `run_config.json`. |

---

## 12. Milestones

| Phase | Outcome | GPU needed |
|---|---|---|
| **P0 — Spec & scaffold** | This PRD + plan + repo scaffold + harness skeleton | No |
| **P1 — Adapter & baseline** | Finance LoRA trained/exported; HF baseline server + correctness gate | Yes (small) |
| **P2 — vLLM** | vLLM single + multi-LoRA, fp16 + AWQ, first knee curves | Yes |
| **P3 — TensorRT-LLM** | fp16 + FP8 engines w/ LoRA; latency anchor | Yes |
| **P4 — NIM** | NIM container + adapter mount; ergonomics + auto-profile numbers | Yes |
| **P5 — Analysis** | Aggregated results, plots, `REPORT.md`, recommendation | No |
| **P6 (future)** | Triton/K8s autoscaling, A100/H100 cross-tier, cost dashboard | Yes |

---

## 13. Open questions

- Exact finance task framing for the adapter (sentiment-as-generation vs short finance-QA) — affects the correctness gate's strictness.
- Cloud provider / instance for the L4 runs (affects the $/1M-token cost model).
- Whether to include **TGI** as a second OSS reference alongside vLLM (out of scope for v1).

---

## 14. References (verify before external citation)

- vLLM — PagedAttention; multi-LoRA serving (`--enable-lora`).
- NVIDIA TensorRT-LLM — `trtllm-build`, LoRA plugin, FP8 via TensorRT Model Optimizer.
- NVIDIA NIM for LLMs — OpenAI-compatible microservice, dynamic multi-LoRA.
- NVIDIA GenAI-Perf — LLM serving benchmark tool (TTFT/ITL/throughput).
- Hu et al., *LoRA*, ICLR 2022 · Lin et al., *AWQ*, MLSys 2024.
- NVIDIA L4 datasheet (Ada Lovelace, 24 GB, FP8) — confirm bandwidth/TFLOPS figures.
