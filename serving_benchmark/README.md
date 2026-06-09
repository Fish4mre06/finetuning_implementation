# Serving Benchmark — LoRA on Llama-3.1-8B across vLLM, TensorRT-LLM, and NVIDIA NIM (L4)

Reusable Solutions-Architect collateral: serve one fine-tuned **finance LoRA adapter**
on **Llama-3.1-8B-Instruct** through a naive baseline and three optimized stacks on a
**single L4 24 GB**, and record **before/after latency + throughput**.

- **What & why:** [`PRD.md`](./PRD.md)
- **How to build it:** [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)

## Status
Spec phase (P0). PRD + implementation plan authored. Serving runs require an **L4 GPU
host** (this is a CPU dev container) — every GPU-dependent step is flagged `[GPU]` in
the plan.

## The one-paragraph pitch
On a constrained, cost-efficient L4, an 8B model in fp16 barely leaves room for KV
cache, so **quantization (FP8 / AWQ-int4) is the headline, not a footnote**. The
benchmark measures TTFT, inter-token latency, p50/p90/p99 end-to-end latency, and
output-token throughput across a concurrency sweep for **baseline → vLLM → TensorRT-LLM
→ NIM**, plus a **multi-LoRA** study (one base, many adapters). Correctness parity is a
blocking gate before any speed number is trusted; every number is a median of ≥3 runs
with dispersion.
