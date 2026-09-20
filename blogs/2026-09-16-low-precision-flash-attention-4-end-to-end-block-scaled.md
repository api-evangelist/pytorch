---
title: "Low Precision Flash Attention 4: End-to-End Block-Scaled Attention for Blackwell"
url: "https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/"
date: "2026-09-16"
author: "Dev (Devashish) Shankar, Darren Liu, Chunzhi Yang, Jackie (Jiaqi) Xu,Markus Hoehnerbach, Jason Xie, Santosh Mohan, Han Xu, Rich Zhu, Josh Fromm, Hongtao Yu, Max Leung, and John Bocharov"
feed_url: "https://pytorch.org/blog/feed/"
---
TL;DR We extend FlashAttention-4 [1] with MXFP8 forward and backward, reaching 2.85 PF/s forward and 2 PF/s backward on LLM shapes. On our internal shapes, FA4 MX8 reaches 2.54 PF/s...
