# Implementation Plan — LoRA Serving Benchmark (vLLM · TensorRT-LLM · NIM on L4)

Companion to [`PRD.md`](./PRD.md). This is the build order, file layout, concrete
commands, and effort estimate. Anything marked **[GPU]** requires the L4 host;
everything else can be authored in this dev container.

---

## 0. Guiding principles

1. **Correctness before speed.** No perf number is recorded for a deployment that
   hasn't passed the output-parity gate against the baseline.
2. **One workload, one harness, all stacks.** Every server exposes an
   OpenAI-compatible endpoint; a single GenAI-Perf-based driver hits them identically.
3. **Median of ≥3 runs, with dispersion.** Single-run numbers are diagnostics, not results.
4. **Pin everything.** Container digests, model revision, adapter hash, driver/CUDA → `run_config.json`.
5. **Document every knob.** Where stacks have no equivalent setting, report both rather than tuning to win.

---

## 1. Proposed repo layout

```
serving_benchmark/
├── PRD.md
├── IMPLEMENTATION_PLAN.md
├── README.md                      # quickstart + host requirements
├── Makefile                       # one-command targets per stack
├── env/
│   ├── requirements-harness.txt   # genai-perf, openai, pandas, matplotlib
│   └── containers.lock            # pinned image digests (vllm, trtllm, nim, triton)
├── adapter/
│   ├── train_finance_lora.py      # produce/export the finance LoRA (rank/alpha/targets)
│   ├── build_prompt_dataset.py    # fixed prompt sets per workload profile
│   └── adapter_card.md            # rank, alpha, target modules, data hash
├── servers/
│   ├── baseline_hf_server.py      # "before": HF generate, OpenAI-compatible shim
│   ├── vllm_launch.sh             # --enable-lora, --max-loras, AWQ option
│   ├── trtllm/
│   │   ├── build_engine.sh        # trtllm-build (fp16) + ModelOpt FP8
│   │   ├── lora_config.md
│   │   └── serve.sh               # trtllm-serve / triton openai endpoint
│   └── nim/
│       ├── run_nim.sh             # NGC container + adapter mount (multi-LoRA)
│       └── nim_notes.md           # auto-selected profile, version capture
├── harness/
│   ├── run_benchmark.py           # GenAI-Perf wrapper: sweep concurrency × workload
│   ├── correctness_gate.py        # greedy-output parity vs baseline
│   ├── schema.py                  # normalized result row (one schema for all stacks)
│   └── aggregate.py               # median + IQR/CI across repeats
├── analysis/
│   ├── plots.py                   # knee curves, VRAM-vs-concurrency, precision, multi-LoRA
│   └── REPORT.md                  # the written deliverable
└── results/                       # gitignored heavy artifacts; CSVs + plots committed
    ├── raw/                       # per-run genai-perf json
    ├── *.csv                      # aggregated
    └── plots/*.png
```

---

## 2. Environment & host requirements **[GPU]**

- **Host:** 1× NVIDIA **L4 24 GB**, recent driver (≥ 550), CUDA 12.x, Docker + NVIDIA Container Toolkit.
- **Access:** HF token for `meta-llama/Llama-3.1-8B-Instruct` (gated) **or** the non-gated fallback (Mistral-7B-Instruct) — harness must run with either.
- **NGC:** API key for NIM + TensorRT-LLM containers.
- **Containers (pin digests in `containers.lock`):** vLLM OpenAI image, `tensorrt_llm` release image, NIM for LLMs image, optionally Triton for GenAI-Perf.
- **This dev container has no GPU** — author code + docs here, run `make` targets on the L4 host.

---

## 3. Build order (phases map to PRD §12)

### P0 — Scaffold & harness skeleton (no GPU)
- Create the layout in §1, the result `schema.py`, and the GenAI-Perf wrapper interface.
- Define the **result row schema** up front so every stack writes the same columns:
  ```
  stack, precision, adapter_count, workload, concurrency,
  ttft_p50, ttft_p90, ttft_p99, itl_p50, itl_p90,
  e2e_p50, e2e_p90, e2e_p99, output_tps, request_tps,
  peak_vram_gb, gpu_util_pct, run_idx, image_digest, model_rev, adapter_hash
  ```
- Write `aggregate.py` (median + IQR across `run_idx`) and `plots.py` against synthetic rows so the analysis path is testable without a GPU (same approach used to validate the training-side CI logic offline).

### P1 — Adapter & baseline **[GPU, small]**
1. `train_finance_lora.py`: LoRA (rank 16–32, alpha 32, target `q_proj,k_proj,v_proj,o_proj`) on a generative finance-sentiment reframe of the repo's data; export PEFT adapter + `adapter_card.md` (+ a merged fp16 variant for the baseline).
2. `build_prompt_dataset.py`: fixed prompt sets for `chat-short` / `RAG-long` / `batch-skew` with controlled input-token lengths.
3. `baseline_hf_server.py`: HF `generate`, single-stream, minimal OpenAI-compatible `/v1/chat/completions` shim.
4. `correctness_gate.py`: capture greedy outputs on the fixed set → this becomes the **reference** all other stacks must match (≥0.99 token agreement, temp 0).

### P2 — vLLM **[GPU]**
1. `vllm_launch.sh`: serve base + `--enable-lora --lora-modules finance=<adapter>`; fp16 first, then AWQ-int4 build.
2. Run `correctness_gate.py` against vLLM → must pass before perf.
3. `run_benchmark.py`: GenAI-Perf concurrency sweep × workloads × 3 repeats.
4. **Multi-LoRA:** load 4 and 16 adapter copies (`--max-loras`), measure overhead vs single.

### P3 — TensorRT-LLM **[GPU]**
1. `trtllm/build_engine.sh`: convert checkpoint → `trtllm-build` with LoRA plugin (fp16); then **FP8** engine via TensorRT Model Optimizer.
2. Serve via `trtllm-serve` (OpenAI-compatible) or Triton; run correctness gate per precision.
3. Benchmark sweep (fp16 + FP8). This is the **latency anchor** and the FP8 concurrency-unlock story.

### P4 — NVIDIA NIM **[GPU]**
1. `nim/run_nim.sh`: launch NIM for the base model, mount the LoRA (dynamic/multi-LoRA), OpenAI endpoint.
2. **Capture the auto-selected optimized profile** (`nim_notes.md`) — this is what makes the NIM-vs-TRT-LLM comparison honest (likely a TRT-LLM/FP8 backend).
3. Correctness gate + benchmark sweep + multi-LoRA. Frame results as **operational ergonomics + auto-profile** vs the hand-built TRT-LLM engine.

### P5 — Analysis (no GPU)
1. `aggregate.py` over all `results/raw/` → committed CSVs.
2. `plots.py`: latency-vs-throughput knee per stack; VRAM-vs-concurrency; fp16/FP8/AWQ precision study; multi-LoRA overhead bars; $/1M-token bar.
3. `REPORT.md`: setup, fairness controls, findings, **per-scenario recommendation** (interactive vs batch vs many-adapter), honest caveats, and the before/after headline.

### P6 — Future (out of v1 scope)
Triton + K8s autoscaling, A100/H100 cross-tier, live cost dashboard.

---

## 4. Priority slice (what to run first if time-boxed)

The full matrix (PRD §8.4) is large. Run in this order so there's a complete story at each checkpoint:

1. **baseline vs vLLM**, fp16, single adapter, `chat-short`, full concurrency sweep → *first before/after knee curve*.
2. Add **TRT-LLM fp16** and **NIM** on the same cell → *4-way stack comparison*.
3. **Precision study** (vLLM AWQ, TRT-LLM/NIM FP8) on `chat-short` → *the L4 quantization headline*.
4. **Multi-LoRA** (1/4/16) on the best stack(s) → *the "one base, many adapters" story*.
5. Repeat the 4-way comparison on **`RAG-long`** → *KV-cache pressure on L4*.

Each checkpoint is independently presentable collateral.

---

## 5. Validation strategy (how we keep it honest)

- **Correctness gate is blocking** (PRD A4): a stack with failing parity is reported as "did not pass," never benchmarked for speed.
- **Dispersion required** (PRD A5): `aggregate.py` refuses to emit a headline if IQR/CI overlaps across the configs being compared — forces "within noise" language instead of a false win.
- **Offline-testable analysis path**: `aggregate.py` + `plots.py` validated on synthetic rows in this container (no GPU) before P1, so the only GPU-blocked work is the actual serving.
- **Reproducibility check**: a second operator can run one `make <stack>` target and regenerate a result CSV within dispersion.

---

## 6. Effort estimate (engineer-days, on an L4 host)

| Phase | Est. | Notes |
|---|---|---|
| P0 scaffold + harness | 1.0 | No GPU; mostly schema + GenAI-Perf wrapper + plotting on synthetic data. |
| P1 adapter + baseline + gate | 1.0 | LoRA train is short; the parity gate is the real work. |
| P2 vLLM | 1.0 | Easiest stack; multi-LoRA included. |
| P3 TensorRT-LLM | 2.0 | Engine build + FP8 quantization is the long pole. |
| P4 NIM | 1.0 | Turnkey, but NGC access + profile capture. |
| P5 analysis + REPORT | 1.0 | Plots + written recommendation. |
| **Total** | **~7 days** | Excludes cloud-instance provisioning + model-access approval. |

---

## 7. Dependencies & gotchas

- **Llama-3.1 gating** can block P1 for days — kick off access on day 0, keep the Mistral-7B fallback wired.
- **TRT-LLM version ↔ container compatibility** is brittle; pin the image and build the engine *inside* that image.
- **FP8 needs Ada/Hopper** — fine on L4, but AWQ-int4 is the vLLM-side quant since vLLM FP8 maturity varies by version; record exact versions.
- **NIM profile auto-selection** may pick a precision you didn't intend — always capture and report it.
- **GenAI-Perf tokenizer** must match the served model or TTFT/ITL token accounting drifts — pass the model's tokenizer explicitly.

---

## 8. Definition of done

- [ ] Adapter + `adapter_card.md` committed; baseline reference outputs captured.
- [ ] All four stacks pass the correctness gate (per precision).
- [ ] Priority-slice matrix (§4) run at ≥3 repeats; CSVs + plots committed.
- [ ] `REPORT.md` with before/after headline, per-scenario recommendation, and caveats.
- [ ] One-command repro per stack; `containers.lock` + `run_config.json` pinned.
