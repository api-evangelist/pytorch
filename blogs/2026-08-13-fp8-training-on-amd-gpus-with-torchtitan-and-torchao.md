---
title: "FP8 Training on AMD GPUs with TorchTitan and TorchAO: Upstreaming Performance Improvements"
url: "https://pytorch.org/blog/fp8-training-on-amd-gpus-with-torchtitan-and-torchao-upstreaming-performance-improvements/"
date: "2026-08-13"
author: "AMD: Rishi Sinha, Yuankai Chen, Liz Li, Shekhar Pandey, Wen Chen, Xiaobo Chen, Yao Fu, Zhenyu Gu, Andy Luo, Peng Sun META: Matthias Reso, Hamid Shojanazeri, TorchAO team, TorchTitan team"
feed_url: "https://pytorch.org/blog/feed/"
---
At the PyTorch Conference 2025, we demonstrated linear scaling beyond 1,000 GPUs on AMD Instinct clusters using Primus-Turbo, an AMD optimization library for training frameworks such as TorchTitan. We have since upstreamed those AMD optimizations so TorchTitan supports AMD Instinct(™) GPUs directly, with competitive FP8 performance out of the box. All contributions mentioned have been merged into upstream pytorch/AO and pytorch/TorchTitan.
