
# vLLM Serving Performance Lab: Tail Latency and Throughput Under Mixed Prompt Lengths (Overall, Short, Long p99)
**Quick links:** [Open in Colab](https://colab.research.google.com/drive/16_GJlYLGUKrZwf0lfQfSVbHuzFugCpjn?usp=sharing) · [Notebook (PDF)](notebook/vllm_serving_project.pdf) · [Artifacts (plots + CSVs)](artifacts/)


## Project Summary

A Colab-driven performance study of the latency–throughput (RPS) tradeoff in vLLM serving under concurrent, mixed prompt-length workloads. Alongside overall p50/p90/p99, it explicitly tracks **short p99 vs long p99 vs overall p99** to quantify head-of-line blocking and mixed-workload interference. The notebook also builds a deliberately **bad/unfair baseline** to demonstrate how poor knob choices can create throughput and tail-latency bottlenecks that **diminish chunked-prefill’s benefits**, before tuning toward a stable fair-scheduling configuration.


## How to run (Colab)

1. Open the Colab notebook and switch runtime to **GPU**.
2. Run cells **top-to-bottom** (or `Runtime → Run all`).
3. The notebook starts/stops a local **vLLM OpenAI-compatible server** (`127.0.0.1:8000`) and runs load sweeps against it.

**Test environment:** Experiments were run on **Google Colab with an NVIDIA L4 GPU**.

**Workloads:** JSONL promptsets (short_only, 90/10, 70/30, 50/50 mixes) are loaded directly by the notebook.

**Outputs:** Each experiment prints summary tables with **RPS + overall/short/long p50/p90/p99** and saves sweep results (CSV) for resume-capable runs.

## Experiment 1 — Mixed Prompt-Length Workloads (Baseline Stress Test)

**Goal:** Establish a baseline for the **latency–throughput tradeoff** under concurrent serving when prompt lengths are mixed. Measure **overall p50/p90/p99 + short p99 vs long p99** and observe interference effects.

**Workloads:** `short_only`, `mix_90_10`, `mix_70_30` (all at fixed concurrency).  
**Metrics:** success/fail, RPS, overall p50/p90/p99 (+ short/long splits tracked elsewhere in the notebook).

### Key results (c = 8, 200 requests)
As the workload includes more long prompts, **tail latency increases and throughput drops**.

- **Throughput (RPS):** `2.53 → 2.29 → 1.96` (short_only → 90/10 → 70/30)
- **Overall p99 latency (s):** `3.59 → 4.53 → 5.38`
- **Overall p50 latency (s):** `3.55 → 3.62 → 4.10`

**Observed failure mode (important):** With fixed `max_tokens=128`, some mixed-workload requests can hit **context/token-limit overflow** (prompt + output > `max-model-len`), causing **HTTP 400 failures**.  
➡️ **Experiment 1.1** fixes this via **dynamic context budgeting**: shrink output tokens first, then trim input (head+tail) only if needed.

### Takeaways
- **Mixed prompt lengths create interference:** long-prefill requests push up **tail latency** for everyone and reduce **effective RPS**.
- **The latency penalty is nonlinear:** moving from 90/10 to 70/30 shows a clear tail expansion (p99 grows faster than p50).
- **Hard failures can appear with naive token settings:** fixed output caps are not safe under variable prompt lengths without budgeting.

### Artifacts
**Summary table:** [`artifacts/exp1/tables/exp01_results_summary.csv`](artifacts/exp1/tables/exp01_results_summary.csv)

**Plots:**
- Latency percentiles by workload:  
  `artifacts/exp1/plots/exp01_latency_percentiles_by_workload.png`  
  ![exp01 latency percentiles](artifacts/exp1/plots/exp01_latency_percentiles_by_workload.png)

- Throughput (RPS) by workload:  
  `artifacts/exp1/plots/exp01_throughput_rps_by_workload.png`  
  ![exp01 throughput](artifacts/exp1/plots/exp01_throughput_rps_by_workload.png)

- Latency histograms:  
  `artifacts/exp1/plots/exp01_latency_hist_short_only.png`  
  `artifacts/exp1/plots/exp01_latency_hist_mix_90_10.png`  
  `artifacts/exp1/plots/exp01_latency_hist_mix_70_30.png`



