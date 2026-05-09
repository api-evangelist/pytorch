---
title: "In-Kernel Broadcast Optimization: Co-Designing Kernels for RecSys Inference"
url: "https://pytorch.org/blog/in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference/"
date: "2026-05-05"
author: "Jian Jiao, Boda Li, Hongtao Yu, Yuanwei (Kevin) Fang, Zhengkai Zhang, Zhuoran Zhao, Yuxin Chen, Sijia Chen†, Yang Chen†, Zijian Shen, Shuyao Bi, Ao Cai, Junhan Hu†, Shuqi Yang†, Wei Wei, Lu Fang, Rengan Xu, Manman Ren, Alex Zhong, Xiaohan Wei, Zeliang Chen, Ellie Wen, Wenlin Chen"
feed_url: "https://pytorch.org/blog/feed/"
---
TL;DR: Traditional RecSys inference explicitly replicates shared user embeddings/sequences for every candidate. In-Kernel Broadcast Optimization (IKBO) eliminates this overhead via a kernel-model-system co-design that fuses broadcast logic directly into user-candidate interaction kernels. By decreasing both the memory footprint and IO utilization, IKBO unlocks even higher throughput. IKBO delivers up to a 2/3 reduction in compute-intensive net latency, serving as the scalability backbone for the request-centric, inference-efficient framework that powers the Meta Adaptive Ranking Model.
