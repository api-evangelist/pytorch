---
title: "Using Muon Optimizer with DeepSpeed"
url: "https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/"
date: "2026-06-03"
author: "Zhipeng Wang, Guokai Ma, Peng Du and Chi McIsaac, DeepSpeed team"
feed_url: "https://pytorch.org/blog/feed/"
---
DeepSpeed now supports the Muon Optimizer, which targets 2D weight matrices in neural networks and maintains only one momentum buffer per parameter compared to Adam's two, resulting in approximately 9% GPU memory savings during fine-tuning. Muon has demonstrated superior performance on benchmarks including MBPP+, MMLU, and GSM8K, and has been adopted by major AI labs including Moonshot AI for training their Kimi-K2 model. The post covers integration details and performance comparisons for teams looking to improve memory efficiency and training convergence.
