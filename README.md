## Project Summary

A Colab-driven performance study of the latency–throughput (RPS) tradeoff in vLLM serving under concurrent, mixed prompt-length workloads. Alongside overall p50/p90/p99, it explicitly tracks **short p99 vs long p99 vs overall p99** to quantify head-of-line blocking and mixed-workload interference. The notebook also builds a deliberately **bad/unfair baseline** to demonstrate how poor knob choices can create throughput and tail-latency bottlenecks that **diminish chunked-prefill’s benefits**, before tuning toward a stable fair-scheduling configuration.


