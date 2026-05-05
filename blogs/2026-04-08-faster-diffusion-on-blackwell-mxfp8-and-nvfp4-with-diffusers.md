---
title: "Faster Diffusion on Blackwell: MXFP8 and NVFP4 with Diffusers and TorchAO"
url: "https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/"
date: "Wed, 08 Apr 2026 16:40:22 +0000"
author: "Vasiliy Kuznetsov (Meta) and Sayak Paul (Hugging Face)"
feed_url: "https://pytorch.org/feed/"
---
<p>Diffusion models for image and video generation have been surging in popularity, delivering super-realistic visual media. However, their adoption is often constrained by the sheer requirements in memory and compute. Quantization is essential for efficient serving of these models.</p>
<p>In this post, we demonstrate reproducible end-to-end inference speedups of up to <strong>1.26x</strong> with MXFP8 and <strong>1.68x</strong> with NVFP4 with <a href="https://github.com/huggingface/diffusers">diffusers</a> and <a href="https://github.com/pytorch/ao">torchao</a> on the <a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">Flux.1-Dev</a>,<a href="https://huggingface.co/Qwen/Qwen-Image"> QwenImage</a>, and<a href="https://huggingface.co/Lightricks/LTX-2"> LTX-2</a> models on NVIDIA B200.  We also outline how we used selective quantization, CUDA Graphs, and LPIPS as a measure to iterate on the accuracy and optimal performance of these models.  The code to reproduce the experiments in this post is <a href="https://github.com/sayakpaul/diffusers-blackwell-quants">here</a>.</p>
<p><img alt="" height="959" src="https://pytorch.org/wp-content/uploads/2026/04/1.png" width="1600" /><strong>Table of contents:</strong></p>
<ul>
<li>Background on MXPF8 and NVFP4</li>
<li>Basic Usage with Diffusers and TorchAO</li>
<li>Benchmark Results</li>
<li>Technical Considerations</li>
</ul>
<h2>Background on MXFP8 and NVFP4</h2>
<p>MXFP8 and NVFP4 are microscaling formats supported natively by NVIDIA’s Blackwell architecture (e.g., B200 GPUs). Unlike standard quantization, which scales an entire tensor, microscaling groups elements into small blocks (e.g., 16 or 32 values) that share a high-precision scale factor. This allows for significantly lower bit-depths while preserving dynamic range and accuracy.</p>
<ul>
<li>MXFP8 (OCP Microscaling FP8): An 8-bit industry-standard format (E4M3/E5M2) from the Open Compute Project (OCP). It uses a block size of 32 with 8-bit scaling. It provides a &#8220;sweet spot&#8221; balance, delivering faster inference than BF16 with virtually no loss in visual quality (lower LPIPS), and often achieves the lowest latency at smaller batch sizes.</li>
<li>NVFP4 (NVIDIA FP4): A 4-bit floating-point format (E2M1) uniquely accelerated by Blackwell Tensor Cores. It uses a block size of 16 with FP8 scaling factors. It offers the highest theoretical throughput and lowest memory footprint (approx. 3.5x smaller than BF16), making it ideal for high-batch, compute-bound workloads.</li>
</ul>
<p>Refer to<a href="https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference/"> this post</a> to know more.</p>
<h2>Basic Usage with diffusers and TorchAO</h2>
<h3>Prerequisites</h3>
<p>NVFP4 requires a CUDA capability of at least 10.0. So, make sure you have a GPU that fits the bill. The benchmarks presented in this document were conducted on a B200 machine (B200 DGX).</p>
<p>For the virtual environment, you can use <code>conda</code>:</p>
<pre><code class="language-sh">conda create -n nvfp4 python=3.11 -y

conda activate nvfp4

pip install --pre torch --index-url
https://download.pytorch.org/whl/nightly/cu130

pip install --pre torchao --index-url
https://download.pytorch.org/whl/nightly/cu130

pip install --pre mslk --index-url
https://download.pytorch.org/whl/nightly/cu130

pip install diffusers transformers accelerate sentencepiece protobuf av imageio-ffmpeg</code></pre>
<p>At the time of writing, the nightlies were <code>2.12.0.dev20260315+cu130</code>, <code>0.17.0.dev20260316+cu130</code>, and <code>2026.3.15+cu130</code> for PyTorch, TorchAO, and MSLK, respectively.</p>
<p>Some models require users to be authenticated on the Hugging Face Hub platform. So, please make sure to run <code>hf auth login</code> before running the examples, if not already done.</p>
<h2>Basic Usage</h2>
<p>Using the NVFP4 quantization config from TorchAO is straightforward with its native integration in Diffusers:</p>
<pre><code>from diffusers import DiffusionPipeline, TorchAoConfig, PipelineQuantizationConfig

import torch

from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig,
)

config = NVFP4DynamicActivationNVFP4WeightConfig(
    use_dynamic_per_tensor_scale=True, use_triton_kernel=True,
)
pipe_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig(config)}
)

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16,
    quantization_config=pipe_quant_config
).to("cuda")
pipe.transformer.compile_repeated_blocks(fullgraph=True)

pipe_call_kwargs = {
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 28,
    "max_sequence_length": 512,
    "num_images_per_prompt": 1,
    "generator": torch.manual_seed(0),
}
result = pipe(**pipe_call_kwargs)
image = result.images[0]
image.save("my_image.png")</code></pre>
<p>The code snippet above quantizes every <code>torch.nn.Linear</code> layer of the model.</p>
<p>For this post, we always use regional compilation with <code>fullgraph=True</code>, as it significantly reduces compilation time and yields results almost as good as full model compilation. Know more about regional compilation from<a href="https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/"> here</a>.</p>
<h3>Recipe Selection</h3>
<p>The code snippet below shows how to configure MXFP8 and NVFP4 inference with TorchAO:</p>
<pre><code># MXFP8

quant_config = MXDynamicActivationMXWeightConfig(
    activation_dtype=torch.float8_e4m3fn,
    weight_dtype=torch.float8_e4m3fn,
    kernel_preference=KernelPreference.AUTO,
)

# NVFP4

quant_config = NVFP4DynamicActivationNVFP4WeightConfig(
    use_dynamic_per_tensor_scale=True,
    use_triton_kernel=True,
)</code></pre>
<h2>Benchmark Results</h2>
<h3>Flux.1-Dev</h3>
<p>The following inference params were used during benchmarking <code>FLUX.1-dev</code>:</p>
<pre><code>{
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 28,
    "max_sequence_length": 512,
}</code></pre>
<h3>Performance and Peak Memory</h3>
<p>First, we present latency and peak memory consumption across different settings and different benchmarks, with speedups up to 1.26x with MXFP8 and up to 1.59x with NVFP4. Note that these results use selective quantization, wherein we exclude certain layers from getting quantized. We discuss more about selective quantization later in this post.</p>
<table width="798">
<thead>
<tr>
<th colspan="5">
<h3>Flux-1.dev performance and peak memory with MXFP8 and NVFP4 quantization</h3>
</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Quant Mode</strong></td>
<td><strong>Batch Size</strong></td>
<td><strong>Latency (s)</strong></td>
<td><strong>Memory (GB)</strong></td>
<td><strong>Speedup vs BF16</strong></td>
</tr>
<tr>
<td>None</td>
<td>1</td>
<td>2.10</td>
<td>38.34</td>
<td>1.00</td>
</tr>
<tr>
<td>MXFP8</td>
<td>1</td>
<td>1.75</td>
<td>26.90</td>
<td>1.21</td>
</tr>
<tr>
<td>NVFP4</td>
<td>1</td>
<td><strong>1.41</strong></td>
<td><strong>21.33</strong></td>
<td>1.50</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>None</td>
<td>4</td>
<td>7.87</td>
<td>44.39</td>
<td>1.00</td>
</tr>
<tr>
<td>MXFP8</td>
<td>4</td>
<td>6.36</td>
<td>32.95</td>
<td>1.24</td>
</tr>
<tr>
<td>NVFP4</td>
<td>4</td>
<td><strong>5.09</strong></td>
<td><strong>27.39</strong></td>
<td>1.55</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>None</td>
<td>8</td>
<td>15.57</td>
<td>53.00</td>
<td>1.00</td>
</tr>
<tr>
<td>MXFP8</td>
<td>8</td>
<td>12.40</td>
<td>41.56</td>
<td>1.26</td>
</tr>
<tr>
<td>NVFP4</td>
<td>8</td>
<td><strong>9.81</strong></td>
<td><strong>36.00</strong></td>
<td>1.59</td>
</tr>
</tbody>
</table>
<p><em>NVIDIA B200, selective quantization, torch.compile with regional compilation; batch_size=1 uses <code>torch.compile(..., mode='reduce-overhead')</code>. Quant Mode &#8220;None&#8221; means no quantization.</em></p>
<h3>Accuracy</h3>
<p>The MXFP8 and NVFP4 images generated for a test prompt are close to the bfloat16 baseline:</p>
<p><img alt="" height="510" src="https://pytorch.org/wp-content/uploads/2026/04/Screenshot-2026-04-01-at-8.08.41-PM.png" width="1286" /></p>
<p>For a more thorough accuracy evaluation, we computed the mean <a href="https://github.com/richzhang/PerceptualSimilarity">LPIPS</a> score between the bfloat16 images (baseline) and MXFP8|NVFP4 images (experiment), averaged over the prompts in the <a href="https://huggingface.co/datasets/sayakpaul/drawbench">Drawbench</a> dataset:</p>
<table width="609">
<thead>
<tr>
<th colspan="2">
<h3>Flux-1.dev mean LPIPS score with MXFP8 and NVFP4 quantization</h3>
</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Quant Mode</strong></td>
<td><strong>Mean LPIPS on Drawbench</strong></td>
</tr>
<tr>
<td>None</td>
<td>0</td>
</tr>
<tr>
<td>MXFP8</td>
<td>0.11</td>
</tr>
<tr>
<td>NVFP4</td>
<td>0.44</td>
</tr>
</tbody>
</table>
<p><em>NVIDIA B200, selective quantization, torch.compile with regional compilation.</em></p>
<p>An LPIPS score of zero means &#8220;identical images&#8221;, and lower LPIPS scores correspond to higher perceptual similarity.  The code we used to compute the mean LPIPS score is <a href="https://github.com/sayakpaul/diffusers-blackwell-quants/blob/5354691e2f171e86245468cbda57af56dd2c606a/README.md?plain=1#L26">here</a>.  Please see the LPIPS section further in this post for more details on accuracy evaluations with LPIPS.</p>
<h3>LTX-2</h3>
<p>For LTX-2, we enabled tiling on the VAE to keep the memory requirements manageable.  The following inference-time parameters were used to obtain the results:</p>
<pre><code> {
        "prompt": (
              "INT. HOME OFFICE - DAY. Soft natural daylight lights a desk with an open laptop. The camera holds a steady medium shot. A small real house cat sits naturally on all fours in front of the laptop, much smaller than the desk and computer. The cat looks at the screen curiously. Suddenly, with a soft magical sparkle effect, a pair of tiny reading glasses appears in midair and gently lands on the cat's face. A faint whimsical chime sound plays. The cat pauses for a split second, then begins pressing the keyboard clumsily with one paw, producing rapid typing sounds. The laptop screen glow reflects softly on the cat's fur while light playful music continues."
        ),
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 768,
        "height": 512,
        "num_frames": 121,
        "frame_rate": 24.0,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
}</code></pre>
<h3>Performance and Peak Memory</h3>
<table width="884">
<thead>
<tr>
<th colspan="5">
<h3>LTX-2 performance and peak memory with MXFP8 and NVFP4 quantization</h3>
</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Quant Mode</strong></td>
<td><strong>Batch Size</strong></td>
<td><strong>Latency (s)</strong></td>
<td><strong>Memory (GB)</strong></td>
<td><strong>Speedup</strong></td>
</tr>
<tr>
<td>None</td>
<td>1</td>
<td>16.230</td>
<td>72.77</td>
<td>1.00</td>
</tr>
<tr>
<td>MXFP8</td>
<td>1</td>
<td>13.724</td>
<td>54.54</td>
<td>1.18</td>
</tr>
<tr>
<td>NVFP4</td>
<td>1</td>
<td><em><strong>10.374</strong></em></td>
<td><em><strong>45.72</strong></em></td>
<td>1.56</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>None</td>
<td>4</td>
<td>61.591</td>
<td>87.61</td>
<td>1.00</td>
</tr>
<tr>
<td>MXFP8</td>
<td>4</td>
<td>50.956</td>
<td>69.38</td>
<td>1.21</td>
</tr>
<tr>
<td>NVFP4</td>
<td>4</td>
<td><em><strong>36.963</strong></em></td>
<td><em><strong>60.56</strong></em></td>
<td>1.67</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>None</td>
<td>8</td>
<td>122.427</td>
<td>107.40</td>
<td>1.00</td>
</tr>
<tr>
<td>MXFP8</td>
<td>8</td>
<td>102.546</td>
<td>89.18</td>
<td>1.19</td>
</tr>
<tr>
<td>NVFP4</td>
<td>8</td>
<td><em><strong>72.689</strong></em></td>
<td><em><strong>80.36</strong></em></td>
<td>1.68</td>
</tr>
</tbody>
</table>
<p><em>NVIDIA B200, selective quantization, torch.compile with regional compilation. Quant Mode &#8220;None&#8221; means no quantization.</em></p>
<h3>Accuracy</h3>
<p>Check out <a href="https://gist.github.com/sayakpaul/ed83f505b6fbed4f4d874826773a891a">this link</a> for a comparison of the video results on a test prompt.  Calculating eval scores over a prompt dataset (like we did for Flux-1.dev) is left for a future study.</p>
<h3>QwenImage</h3>
<p>The following inference-time parameters were used to obtain the results:</p>
<pre><code> {
    "prompt": "A cat holding a sign that says hello world",
    "negative_prompt": " ",
    "height": 1024,
    "width": 1024,
    "true_cfg_scale": 4.0,
    "num_inference_steps": 50,
}</code></pre>
<h3>Performance and Peak Memory</h3>
<table width="723">
<thead>
<tr>
<th colspan="5">
<h3>QwenImage performance and peak memory with MXFP8 and NVFP4 quantization</h3>
</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Quant Mode</strong></td>
<td><strong>Batch Size</strong></td>
<td><strong>Latency (s)</strong></td>
<td><strong>Memory (GB)</strong></td>
<td><strong>Speedup</strong></td>
</tr>
<tr>
<td>None</td>
<td>1</td>
<td>7.454</td>
<td>62.21</td>
<td>1.00</td>
</tr>
<tr>
<td>MXFP8</td>
<td>1</td>
<td>6.430</td>
<td>55.65</td>
<td>1.16</td>
</tr>
<tr>
<td>NVFP4</td>
<td>1</td>
<td><em><strong>5.369</strong></em></td>
<td><em><strong>52.45</strong></em></td>
<td>1.39</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>None</td>
<td>4</td>
<td>26.779</td>
<td>75.52</td>
<td>1.00</td>
</tr>
<tr>
<td>MXFP8</td>
<td>4</td>
<td>21.835</td>
<td>68.97</td>
<td>1.23</td>
</tr>
<tr>
<td>NVFP4</td>
<td>4</td>
<td><em><strong>18.279</strong></em></td>
<td><em><strong>65.76</strong></em></td>
<td>1.47</td>
</tr>
<tr>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>None</td>
<td>8</td>
<td>52.095</td>
<td>92.47</td>
<td>1.00</td>
</tr>
<tr>
<td>MXFP8</td>
<td>8</td>
<td>41.569</td>
<td>85.91</td>
<td>1.25</td>
</tr>
<tr>
<td>NVFP4</td>
<td>8</td>
<td><em><strong>34.969</strong></em></td>
<td><em><strong>82.7</strong></em></td>
<td>1.49</td>
</tr>
</tbody>
</table>
<p><em>NVIDIA B200, selective quantization, torch.compile with regional compilation, batch_size=1 uses <code>torch.compile(..., mode='reduce-overhead')</code>. Quant Mode &#8220;None&#8221; means no quantization.</em></p>
<h3>Accuracy</h3>
<p>The MXFP8 and NVFP4 images generated for a test prompt are close to the bfloat16 baseline, with NVFP4 showing slightly larger differences vs MXFP8:</p>
<p><img alt="" height="534" src="https://pytorch.org/wp-content/uploads/2026/04/Screenshot-2026-04-01-at-8.15.10-PM.png" width="1304" /></p>
<p>In the following table, we report the LPIPS scores similar to Flux.1-Dev.</p>
<table width="683">
<thead>
<tr>
<th colspan="2">
<h3>QwenImage mean LPIPS score with MXFP8 and NVFP4 quantization</h3>
</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Quant Mode</strong></td>
<td><strong>Mean LPIPS on Drawbench</strong></td>
</tr>
<tr>
<td>None</td>
<td>0</td>
</tr>
<tr>
<td>MXFP8</td>
<td>0.34</td>
</tr>
<tr>
<td>NVFP4</td>
<td>0.41</td>
</tr>
</tbody>
</table>
<p>Note: In our experiments, we found QwenImage to be more sensitive to quantization than Flux.1-Dev, as evidenced by the higher mean MXFP8 LPIPS score of 0.34 for QwenImage (compared to a mean LPIPS score of 0.11 for MXP8 on Flux-1.Dev).  Reducing the mean LPIPS score for QwenImage further via more aggressive selective quantization or more advanced numerical algorithms (GPTQ, QAT, etc) is left for a future study.</p>
<h2>Technical Considerations</h2>
<p>In this section, we share how we used selective quantization, CUDA Graphs, and LPIPS to iterate on the performance and accuracy metrics presented in this post.</p>
<h2>Optimizing Accuracy and Performance with Selective Quantization</h2>
<p>We used selective quantization to optimize for latency (all models) and LPIPS (Flux-1.dev), skipping layers based on two simple heuristics:</p>
<ol>
<li>If the weight or activation shape of a <code>torch.nn.Linear</code> is too small to benefit from quantization <code>min(M, K, N) &lt; 1024)</code>, skip it.  This is to ensure that the speedup from quantizing the matrix multiply is larger than the additional overhead of quantizing the activation (more context: <a href="https://docs.pytorch.org/ao/main/workflows/inference.html#microbenchmarks-and-roofline-model">here</a>).
<ul>
<li>A tutorial for how to find the weight and activation shapes in your model using <code>torchao</code> tooling is<a href="https://docs.pytorch.org/ao/main/eager_tutorials/debugging_weights_and_activations.html"> here</a>. Note that even if the weight is large, a small activation shape could make quantization not profitable.</li>
</ul>
</li>
<li>If the layer is likely to meaningfully contribute to model accuracy (such as embeddings, normalization), skip it.
<ul>
<li>To apply this on your model, you can print out the model (<code>print(model)</code>) and inspect the FQNs manually, then skip the FQNs you suspect could be impacting accuracy based on your knowledge of the model architecture.</li>
</ul>
</li>
</ol>
<p>The exact heuristics we used for each model are:</p>
<ol>
<li><a href="https://github.com/sayakpaul/diffusers-blackwell-quants/blob/f313fe7dcb44f55dae4dd5191239bad15fa2a5b6/benchmark.py#L190-L201">Flux-1.dev</a></li>
<li><a href="https://github.com/sayakpaul/diffusers-blackwell-quants/blob/fd427a86f53e46f2511ddaf65759a59b86d6ceb1/benchmark.py#L137">QwenImage</a></li>
<li><a href="https://github.com/sayakpaul/diffusers-blackwell-quants/blob/fd427a86f53e46f2511ddaf65759a59b86d6ceb1/benchmark.py#L160">LTX-2</a></li>
</ol>
<p>To quantify the impact of selective quantization, we measure performance, memory, and mean<a href="https://github.com/richzhang/PerceptualSimilarity"> LPIPS</a> (with AlexNet) between the images with pure Bfloat16 and images generated with NVFP4 and MXFP8.</p>
<table width="566">
<thead>
<tr>
<th colspan="4">
<h3>Impact of full vs selective quantization on Flux-1.dev</h3>
</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Quant Mode</strong></td>
<td><strong>LPIPS</strong></td>
<td><strong>Latency (s)</strong></td>
<td><strong>Memory (GB)</strong></td>
</tr>
<tr>
<td>MXFP8 + full quantization</td>
<td>0.138128</td>
<td>1.774</td>
<td>26.84</td>
</tr>
<tr>
<td>MXFP8 + selective quantization</td>
<td><em><strong>0.107562</strong></em></td>
<td>1.746</td>
<td>26.90</td>
</tr>
<tr>
<td>NVFP4 + full quantization</td>
<td>0.479679</td>
<td>2.112</td>
<td>21.25</td>
</tr>
<tr>
<td>NVFP4 + selective quantization</td>
<td><em><strong>0.438337</strong></em></td>
<td>2.076</td>
<td>21.33</td>
</tr>
</tbody>
</table>
<p><em>(Lower LPIPS is better, with LPIPS of ~0.1 usually meaning that the images are nearly indistinguishable. LPIPS computation code is available<a href="https://github.com/sayakpaul/diffusers-blackwell-quants/blob/f313fe7dcb44f55dae4dd5191239bad15fa2a5b6/compute_lpips.py"> here</a>).</em></p>
<p>As we can notice from the results above, excluding certain layers from quantization (aka “selective quantization”) provides the best trade-off between latency, peak memory consumption, and LPIPS. Therefore, we follow the recipe of selective quantization for the rest of the two models reported in this post.</p>
<p>We used simple heuristics to find our selective quantization recipes. There are more advanced approaches for selective quantization, such as<a href="https://huggingface.co/blog/badaoui/sensitivity-aware-mixed-precision-quantizer-v1#layer-sensitivity-estimation"> this layer sensitivity study</a>.</p>
<p>Note that while iterating on our selective quantization recipes, we found performance gaps in TorchAO’s kernel for quantizing tensors to NVFP4. We improved NVFP4 performance in <a href="https://github.com/pytorch/ao/pull/4031">this PR</a> by upgrading the `to_nvfp4` kernel to use <a href="https://github.com/meta-pytorch/MSLK">MSLK</a>.</p>
<h3>Improving CPU Overhead with CUDA Graphs</h3>
<p>We noticed that when using NVFP4 with small batch sizes like 1, CPU overhead tends to have a nontrivial impact on latency improvements. To significantly reduce this overhead, we used the “reduce-overhead” compilation mode, which enables CUDA graphs. Below, we provide the profile traces before and after applying CUDA Graphs.</p>
<p><img alt="" height="1064" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed.png" width="1578" /></p>
<p>To cleanly compose <code>torch.compile(..., mode='reduce-overhead')</code> with the per-block compilation from the <code>diffusers</code> library, we had to wrap each transformer block in a function that clones its inputs.  The PR to do this is <a href="https://github.com/sayakpaul/diffusers-blackwell-quants/pull/1">here</a>, showing a 1.81x speedup for <code>QwenImage + nvfp4</code> at <code>batch_size==1</code>.</p>
<h3>Evaluating Image Generation Accuracy with LPIPS</h3>
<p>We used the LPIPS (<a href="https://github.com/richzhang/PerceptualSimilarity">GitHub</a>) metric to compare how similar images generated by a quantized model are from the images generated by the baseline (bfloat16) model. In pseudocode:</p>
<pre><code>lpips_scores = []

for text_prompt in dataset:
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = {"prompt": prompt, "generator": generator, ...}
    image_baseline = pipe_bf16(**kwargs)
    image_quantized = pipe_quantized(**kwargs)
    lpips_score = calculate_lpips_score(image_baseline, image_quantized)
    lpips_scores.append(lpips_score)

lpips_mean = lpips_scores.sum() / len(lpips_scores)</code></pre>
<p>The actual code we used is <a href="https://github.com/sayakpaul/diffusers-blackwell-quants/blob/main/compute_lpips.py">here</a>.</p>
<h3>Example LPIPS Scores for Pairs of Images</h3>
<p>This section provides example LPIPS scores for pairs of images to help put the LPIPS metrics reported above into context, and enable readers to reason about “what is a good LPIPS score”.</p>
<p>The images below were generated with <code>FLUX.1-dev</code>. The images on the left are the baseline (bfloat16), and the images on the right are from quantizing every <code>torch.nn.Linear</code> of the model with MXFP8. The LPIPS scores are based on the comparison of the image on the right (experiment) to the image on the left (baseline).</p>
<p><img alt="" height="615" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-1.png" width="1068" /></p>
<p><img alt="" height="613" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-2.png" width="1065" /></p>
<p><img alt="" height="611" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-3.png" width="1060" /></p>
<p>Below, we provide a similar comparison but with NVFP4 images on the right-hand side.</p>
<p><img alt="" height="606" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-4.png" width="1051" /></p>
<p><img alt="" height="607" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-5.png" width="1054" /></p>
<h2>Conclusion</h2>
<p>In this post, we investigated the performance of NVFP4 and MXFP8 quantization schemes on popular image and video generation models. We presented the recipes that provide a reasonable trade-off between speed, quality, and memory. We also uncovered some important issues that can get in the way of optimal performance and how we can approach them. We hope these recipes will help improve the performance of your image and video generation workloads.</p>
<h2>Resources</h2>
<ul>
<li><a href="https://github.com/sayakpaul/diffusers-blackwell-quants">Code repository</a></li>
<li>TorchAO docs:
<ul>
<li><a href="https://docs.pytorch.org/ao/main/api_reference/generated/torchao.prototype.mx_formats.MXDynamicActivationMXWeightConfig.html">MXFP8</a></li>
<li><a href="https://docs.pytorch.org/ao/main/api_reference/generated/torchao.prototype.mx_formats.NVFP4DynamicActivationNVFP4WeightConfig.html">NVFP4</a></li>
</ul>
</li>
<li>Diffusers x TorchAO<a href="https://huggingface.co/docs/diffusers/main/en/quantization/torchao"> integration</a></li>
</ul>
<p>All outputs can be found<a href="https://huggingface.co/datasets/sayakpaul/diffusers-blackwell-quants"> here</a><span style="font-weight: 400;"><br />
</span></p>
