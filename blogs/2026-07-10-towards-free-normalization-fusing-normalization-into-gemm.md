---
title: "Towards Free Normalization: Fusing Normalization into GEMM and Attention Kernels"
url: "https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/"
date: "2026-07-10"
author: "Jacky (Junqing) Zhou, Hongtao Yu, Jackie (Jiaqi) Xu, Menglu Yu, Ethan Che, Han Xu, Darren Liu, Peng Chen (Dev Infra), Daohang Shi, Max Leung"
feed_url: "https://pytorch.org/blog/feed/"
---
Code available at: https://github.com/facebookresearch/ads_model_kernel_library/tree/main/multi_cta_norm_fusion and https://github.com/facebookresearch/ads_model_kernel_library/tree/main/gdpa_megakernel TL;DR In this blog post, we present various novel kernel fusion techniques for common normalization ops like LayerNorm and RMSNorm, which provide significant speedup...
