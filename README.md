# vllm-serving-performance-lab
A Colab-driven performance study of vLLM serving under mixed prompt lengths. Benchmarks throughput (RPS) vs tail latency (p50/p90/p99) across concurrency, “bad” vs fair scheduling configurations, and key server knobs (max num batched tokens, max num seqs), with sweep scripts and plots.
