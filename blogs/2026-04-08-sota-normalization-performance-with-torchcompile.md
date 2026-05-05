---
title: "SOTA Normalization Performance with torch.compile"
url: "https://pytorch.org/blog/sota-normalization-performance-with-torch-compile/"
date: "Wed, 08 Apr 2026 07:00:02 +0000"
author: "Shunting Zhang, Paul Zhang, Elias Ellison, Markus Hoehnerbach, Jason Ansel, Natalia Gimelshein"
feed_url: "https://pytorch.org/feed/"
---
<h2><span style="font-weight: 400;">Introduction</span></h2>
<p><span style="font-weight: 400;">Normalization methods (LayerNorm/RMSNorm) are foundational in deep learning and are used to normalize values of inputs to result in a smoother training process for deep learning models. We evaluate and improve torch.compile performance for LayerNorm/RMSNorm on NVIDIA H100 and B200 to reach near SOTA performance on a kernel-by-kernel basis, in addition with further speedups through automatic fusion capabilities. </span></p>
<h2><span style="font-weight: 400;">Forwards</span></h2>
<p><strong>LayerNorm</strong></p>
<p><span style="font-weight: 400;">LayerNorm was first introduced in this paper: </span><a href="https://arxiv.org/abs/1607.06450"><span style="font-weight: 400;">https://arxiv.org/abs/1607.06450</span></a><span style="font-weight: 400;">. It normalizes the inputs by taking the mean and variance, along with scaling by learnable parameters, gamma (weight) and Beta (bias). </span></p>
<p><img alt="" class="aligncenter wp-image-62295 size-full" height="180" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-17.png" width="498" /></p>
<p><strong>RMSNorm</strong></p>
<p><span style="font-weight: 400;">RMSNorm (root mean square norm) was introduced as a follow up of LayerNorm in this paper: </span><a href="https://arxiv.org/abs/1910.07467"><span style="font-weight: 400;">https://arxiv.org/abs/1910.07467</span></a><span style="font-weight: 400;">. Instead of centering on the mean, the RMS is used to normalize, which is a sum of the squares of x values. We still use gamma (weight) as a learnable parameter for scaling, although there is no longer a bias term.</span></p>
<p><img alt="" class="aligncenter wp-image-62298 size-full" height="180" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-18.png" width="1036" /></p>
<p><span style="font-weight: 400;">The forward pass for both LayerNorm and RMSNorm are relatively similar, typically with a reduction across the contiguous dimension and some extra pointwise ops, with RMSNorm typically being a bit more efficient as there are fewer flops and no bias. For the purposes of this study, we present benchmark results among LayerNorm and RMSNorm interchangeably given the similarity of the kernels.</span></p>
<p><strong>Quack</strong></p>
<p><span style="font-weight: 400;">Quack is a library of hyper optimized CuteDSL kernels from Tri Dao: </span><a href="https://github.com/Dao-AILab/quack"><span style="font-weight: 400;">https://github.com/Dao-AILab/quack</span></a><span style="font-weight: 400;">. Their current README shows on H100 how Quack outperforms torch.compile for these reduction kernels. We use Quack as the SOTA baseline of which we evaluate the performance of torch.compile on. Quack’s README showcases previous results from torch.compile performance below, of which it can be observed that torch.compile ~50% of Quack performance typically.</span></p>
<p><img alt="" class="aligncenter wp-image-62299 size-full" height="466" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-19.png" width="1600" /></p>
<p><strong>torch.compile</strong></p>
<p><span style="font-weight: 400;">Below we illustrate the general logic of a torch.compile generated kernel for LayerNorm forwards, with the same approach for RMSNorm). We assume that the input reduction dimension (rnumel) is contiguous, which we refer to in Inductor as an </span><b><i>Inner reduction</i></b><span style="font-weight: 400;">.</span></p>
<p><img alt="" class="aligncenter wp-image-62300 size-full" height="948" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-20.png" width="1600" /></p>
<p><span style="font-weight: 400;">While the kernel might look a bit confusing, what’s actually happening is very simple:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Maintain partial sums of size R_BLOCK for each row in X the input</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Use partial sums to calculate mean and variance</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Apply elementwise to X based on layernorm formula</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Store output of elementwise</span>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Store mean and variance if elementwise_affine=True and requires_grad=True for backwards</span></li>
</ul>
</li>
</ul>
<p><span style="font-weight: 400;">As a side note, if R is smaller than some heuristic (1024), then Inductor generates a </span><b><i>persistent reduction</i></b><span style="font-weight: 400;">, where we no longer need to loop over the r dimension. Instead, we go directly to taking the mean.</span></p>
<p><span style="font-weight: 400;">In comparing the torch.compile vs Quack versions of RMSNorm forwards, we can reproduce the poor performance of torch.compile compared to Quack on H100 and B200. However, after autotuning and using that to motivate Inductor defaults, we arrive at SOTA performance on H100 and B200. In general, the following was done to achieve this result:</span></p>
<ul>
<li><b>Inserting torch._dynamo.reset() during benchmarking</b><span style="font-weight: 400;"> &#8211; makes sure that torch.compile does not use automatic dynamic shapes, as previously a torch.compile call per shape was performed, making the compiler assume dynamic shapes</span></li>
<li><b>Poor Autotune Configuration Decisions</b><span style="font-weight: 400;"> &#8211; By default was making suboptimal decisions for the autotune configs on H100 and B200, leading to poor performance, though this is mitigated with mode=’max-autotune’. Several improvements were made to the default heuristics: </span>
<ul>
<li><b>Scale up inner reduction RBLOCK</b></li>
<li><b>Scale XBLOCK in persistent reductions </b><span style="font-weight: 400;">for smaller reductions, numel &lt;= 2048</span></li>
<li><b>Decrease num_warps</b><span style="font-weight: 400;"> based on certain reduction dimensions. Num_warps would often be too large for peak vectorization. Peak vectorization is essential for maximizing bytes in flight -&gt; saturating peak memory bandwidth for memory bound workloads, of which Blackwell is more sensitive to given the higher memory bandwidth. </span></li>
</ul>
</li>
</ul>
<p><strong>Benchmark Results</strong></p>
<p><span style="font-weight: 400;">Below we present benchmark results of torch.compile 2.11 vs Quack (March 24th 2026 trunk) on the Quack benchmark shapes alongside some common shapes in the wild, with large M, small N. We demonstrate that torch.compile is generally on parity with Quack. There are two classes of regressions that do occur:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Small regressions on N=384, as Triton is unable to cleanly represent non power-of-2 block size</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Large regressions on very large N for H100, due to the inability to represent distributed shared memory in Triton</span></li>
</ul>
<p><img alt="" class="aligncenter wp-image-62639 size-full" height="836" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-29.png" width="1380" /></p>
<p><strong><img alt="" class="aligncenter wp-image-62644 size-full" height="742" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-1-1.png" width="1380" /></strong></p>
<p><strong>Backwards</strong></p>
<p><span style="font-weight: 400;">The backwards pass for LayerNorm/RMSNorm is a bit more involved than the forward pass. We have to calculate at least 2 gradients, dX for the input, dW for the weights, and optionally dB for the bias in LayerNorm. To simplify and avoid the associated complex math formulas, for performance considerations, these gradient calculations require reductions across both dimensions of dY, the incoming gradient to the backwards pass (the gradient of the previous output in the forwards).</span></p>
<p><span style="font-weight: 400;">The naive option here, and what is sometimes unavoidable with a very large reduction dimension, is to perform the reductions in separate kernels, one for dX, and one for dW, dB. However, that leads to reading the same inputs (dY) in 2 separate kernels, doubling the bytes being read. Given the memory bound nature of normalization kernels, leads to significant additional latency. </span></p>
<p><img alt="" class="aligncenter wp-image-62305 size-full" height="1408" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-23.png" width="1600" /></p>
<p><strong>Fused Reductions</strong></p>
<p><span style="font-weight: 400;">For reasonable shapes where numel is generally not too large and a single row can fit adequately in a thread block, generally &lt;= 16384, it is possible to have a more performant fused kernel that doesn’t blow up shared memory/registers. Essentially, the kernel would perform the reduction for dW, dB as normal but for each row also reduce the columns for dX at the same time. Existing literature exists for this type of fusion, such as in </span><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py#L459"><span style="font-weight: 400;">Liger</span></a><span style="font-weight: 400;">, </span><a href="https://fb.workplace.com/groups/257735836456307/posts/999376505625566"><span style="font-weight: 400;">a fused semi-persistent normalization backwards from Meta</span></a><span style="font-weight: 400;">, and </span><a href="https://github.com/Dao-AILab/quack/blob/main/quack/rmsnorm.py"><span style="font-weight: 400;">Quack’s fused kernel in CuteDSL</span></a><span style="font-weight: 400;">.</span></p>
<p><span style="font-weight: 400;">In Inductor, we represent reductions with distinct types, such as:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">INNER reduction: reductions that reduce thru the stride=1 dimension</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">OUTER reduction: reductions that reduce thru the remaining dimension</span></li>
</ul>
<p><span style="font-weight: 400;">Based on these definitions, the fused kernel is an INNER and OUTER reduction on the same input tensor, with the INNER reduction as dX (contiguous) and the OUTER reduction as (dW, dB). </span></p>
<h4><span style="font-weight: 400;">Split Reduction</span></h4>
<p><span style="font-weight: 400;">Typically for many shapes in the wild, xnumel or the batch dimension is large, much larger than rnumel. In this case, it is generally preferred to process partial sums of the reduction across X and a final torch.sum of the partial sums to allow for better parallelism. The Triton tutorial layernorm illustrates the split reduction, though they utilize locks with atomics with a single thread-block being responsible for individual rows, which is poor for performance on a larger batch dimension (X) and </span><b>leads to numerical inconsistencies</b><span style="font-weight: 400;">:</span><span style="font-weight: 400;"><br />
</span></p>
<p><img alt="" class="aligncenter wp-image-62310 size-full" height="515" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-24.png" width="1232" /></p>
<p><span style="font-weight: 400;">Inductor has similar capabilities currently with split reduction, which allocates a workspace tensor for the partial sums, like above, but does not use atomics, instead ensuring that a single CTA processes multiple rows and writes to one unique spot in the workspace tensor.</span></p>
<p><strong>Inductor Generated Fused Norm Backwards</strong></p>
<p><span style="font-weight: 400;">Combining the fused and split reduction paradigms described above, we enable TorchInductor to automatically generate fused state-of-the-art normalization backward kernels. Furthermore, allowing the compiler to generate such kernels allows for more autotuning and </span><b>automatic fusion capabilities with surrounding operations</b><span style="font-weight: 400;">. Since the main challenge here is to fuse reductions with the same input but different reduction order, we call this optimization MixOrderReduction.</span></p>
<p><span style="font-weight: 400;">For a given [M, N] shape input, the generated kernel performs:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Split-reduction by splitting the M dimension with SPLIT_SIZE chunks</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">for each chunk, we have one row in the workspace tensor saving the partial reduced results for the OUTER reduction (e.g. partial sum of each column or dW, dB)</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">for each chunk, we want to load each row in the chunk by a loop</span>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">do the INNER reduction as usual (e.g. sum the entire row or dX)</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Combine the loaded row with the row in the workspace tensor as the updated partial reduced result</span></li>
</ul>
</li>
</ul>
<p><span style="font-weight: 400;">We have an extra reduction to reduce the partial reduced results in the workspace tensor to get the final result for the OUTER reduction. The extra kernel works on much smaller input tensors so it’s not a huge performance hit to have it in a separate kernel.</span></p>
<p><span style="font-weight: 400;">In the Inductor codegen logic itself, we perform the following steps after recognizing the mix order reduction pattern:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">for the OUTER reduction kernel, we replace the reduction and store_reduction nodes with a new type of partial_accumulate node. This node tracks the value being reduced, what kind of reduction we do etc. This transformation converts the OUTER reduction kernel into a pointwise kernel (PW1)</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Reorder loops for the transformed pointwise kernel (PW1) leveraging the previous loop reordering work and we get (PW2)</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Now PW2 and the INNER reduction have the same loop order and we can fuse them</span></li>
</ul>
<p><strong>Autotuning for Split-Size</strong></p>
<p><span style="font-weight: 400;">SPLIT_SIZE is very critical to the perf of mix-order reduction kernels. The default perf of the Liger RMSNorm backwards kernel on shape (1152000, 384) with dtype=bfloat16 achieves 0.417 TB/s on H100. When reducing the SPLIT_SIZE by 32x, we get 1.912 TB/s.</span></p>
<p><span style="font-weight: 400;">We demonstrate results across the shapes we benchmark and different split sizes on H100 for torch.bfloat16 dtype.</span></p>
<p><img alt="" class="aligncenter wp-image-62311 size-full" height="778" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-25.png" width="1600" /></p>
<p><span style="font-weight: 400;">As shown above, we can conclude that:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">An improper split-size choice can cause &gt; 2x perf degradation</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">The curve is more or less a parabola shape. An autotuning strategy to keep expanding to 2x or 1/2 split size until we found a maximum should be a very effective strategy for this problem. </span></li>
</ul>
<p><span style="font-weight: 400;">Inductor&#8217;s existing split-reduction feature may split the outer reduction for better perf. The split size picked by split-reduction (shown as &#8216;fused_split_reduction&#8217; column in the chart) may be bad due to using an unrelated heuristics. We make MixOrderReduction ignore the split size picked for split-reduction and use its own heuristics or autotuning mechanism to pick a better split-size.</span></p>
<p><strong>Software Pipelining</strong></p>
<p><span style="font-weight: 400;">Another discovery while trying to achieve peak bandwidth on the backwards kernel is the addition of software pipelining, aka prefetching loads. Typically, only compute intensive workloads like GEMM and Attention performed pipelining as more memory bound workloads did not need it, with no num_stages autotuning in Inductor for pointwise/reduction kernels or the Liger examples. However, we observed that in the Quack kernels there was some notion of prefetching. We added num_stages as an autotuning parameter for Inductor kernels generally, and saw significant speedups for some shapes, especially for large M, small N, up to 20% when applied to MixOrderReduction:</span></p>
<p><img alt="" class="aligncenter wp-image-62312 size-full" height="842" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-26.png" width="1362" /></p>
<p><strong>Benchmark Results</strong></p>
<p><span style="font-weight: 400;">Below we present benchmark results for MixOrderReduction compared to PyTorch eager and previous compile, alongside OSS baselines such as Quack and liger. Both of these benchmarks were run on a 750W B200 machine on CUDA 12.9 in late 2025.</span></p>
<p><img alt="" class="aligncenter wp-image-62315 size-full" height="1063" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-27.png" width="1600" /></p>
<p><span style="font-weight: 400;">We observe that:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">The torch.compile w/ MixOrderReduction is 17.07x faster than eager, while torch.compile w/o MixOrderReduction is only 9.93x faster than eager.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">We observe the torch.compile w/ MixOrderReduction is 1.45x faster than Liger and 1.34x faster than Quack</span></li>
</ul>
<p><span style="font-weight: 400;">We also present benchmarking results for LayerNorm, expecting similar results to RMSNorm due to the similarity in the kernels.</span></p>
<p><img alt="" class="aligncenter wp-image-62316 size-full" height="1030" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-28.png" width="1600" /></p>
<p><span style="font-weight: 400;">We observe the same trend in the results as RMSNorm, where torch.compile w/o MixOrderReduction has a significant speedup compared to PyTorch eager. However, torch.compile w/ the new MixOrderReduction paradigm has almost a 2x speedup compared to the previous torch.compile baseline, much closer to peak memory bandwidth.</span></p>
<h2><span style="font-weight: 400;">Conclusion</span></h2>
<p><span style="font-weight: 400;">We improved torch.compile to generate near SOTA forward and backward normalization kernels on H100 and B200 through torch.compile on standard shapes compared to Quack. On top of these optimized kernels, torch.compile provides automatic fusion capabilities of surrounding ops, other pointwise/reductions, allowing for better e2e performance than hand authored kernels. </span></p>
