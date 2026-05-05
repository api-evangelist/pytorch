---
title: "PyTorch Conference Europe 2026: A Landmark Moment for Open Source AI in Paris"
url: "https://pytorch.org/blog/pytorch-conference-europe-2026-a-landmark-moment-for-open-source-ai-in-paris/"
date: "Wed, 15 Apr 2026 22:02:07 +0000"
author: "PyTorch Foundation"
feed_url: "https://pytorch.org/feed/"
---
<p><span style="font-weight: 400;">The first-ever PyTorch Conference Europe April 7-8, 2026 brought together more than 600 researchers, developers, practitioners, and academics in Paris for two packed days of keynotes, technical deep dives, lightning talks, poster sessions, and community connection. From bare-metal infrastructure to agentic AI, the sessions spanned the full AI stack and made one thing clear: the open source AI ecosystem is accelerating faster than ever.</span></p>
<p><span style="font-weight: 400;">All sessions recordings will be available on our</span><a href="https://www.youtube.com/pytorch"> <span style="font-weight: 400;">YouTube channel</span></a><span style="font-weight: 400;"> within the next week. Here is our recap of conference highlights.</span></p>
<p><b><img alt="" class="aligncenter wp-image-66127 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55205663499_9ba0d5d5e0_o-1.jpg" width="2400" /></b></p>
<p><b>Major Announcements:</b></p>
<p><span style="font-weight: 400;">During PyTorchCon EU, the PyTorch Foundation announced new projects joining its community alongside PyTorch, vLLM, DeepSpeed, and Ray. Both Helion and Safetensors have now joined as foundation-hosted projects too. ExecuTorch also became a part of PyTorch Core.</span></p>
<ul>
<li style="font-weight: 400;"><b>Helion</b><span style="font-weight: 400;">, contributed by Meta, is a Python-embedded domain-specific language (DSL) that makes it easy to write fast, scalable ML kernels with minimal boilerplate. By making kernel authoring a first-class part of PyTorch, Helion strengthens custom kernel creation and reduces manual coding effort through autotuning.</span><a href="https://pytorch.org/blog/pytorch-foundation-welcomes-helion-as-a-foundation-hosted-project-to-standardize-open-portable-and-accessible-ai-kernel-authoring/"> <span style="font-weight: 400;">Read more.</span></a></li>
<li style="font-weight: 400;"><b>Safetensors</b><span style="font-weight: 400;">, contributed by Hugging Face, is a secure model file format that prevents arbitrary code execution and enhances performance across multi-GPU and multi-node deployments. It has become one of the most widely used metadata formats for model distribution.</span><a href="https://pytorch.org/blog/pytorch-foundation-announces-safetensors-as-newest-contributed-project-to-secure-ai-model-execution/"> <span style="font-weight: 400;">Read more.</span></a></li>
</ul>
<p><b>ExecuTorch</b><span style="font-weight: 400;"> was officially welcomed as a part of PyTorch Core. Originally developed by Meta, ExecuTorch simplifies running PyTorch models on edge and on-device environments, including mobile phones, AR/VR headsets, and microcontrollers. It was designed around four core principles: end-to-end developer experience, portability across hardware, small and modular efficiency, and openness by default.</span><a href="https://pytorch.org/blog/executorch-becomes-part-of-pytorch-core/"> <span style="font-weight: 400;">Read more.</span></a></p>
<p><span style="font-weight: 400;">These additions reinforce the Foundation&#8217;s position as the vendor-neutral hub for the open source AI stack, covering the full lifecycle from training through inference.</span></p>
<p><b><img alt="" class="aligncenter wp-image-66115 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55205606673_eedf6673e5_o.jpg" width="2400" /></b></p>
<p><b>Day 1: Tuesday, April 7</b></p>
<p><b>Opening Keynotes</b></p>
<p><span style="font-weight: 400;">The morning opened on the Master Stage with a series of keynotes that set the tone for the entire event.</span></p>
<p><b>Mark Collier</b><span style="font-weight: 400;">, Executive Director of the PyTorch Foundation and General Manager of AI and Infrastructure at the Linux Foundation, kicked things off with &#8220;Co-Evolution: How the Open Source Intelligence Stack Compounds.&#8221; His message was clear: the power of the PyTorch ecosystem comes not from any single project, but from how its many components reinforce and build on each other. As the Foundation continues to grow its project portfolio, that compounding effect is only getting stronger.</span></p>
<p><b>Edward Yang</b><span style="font-weight: 400;">, Research Engineer at Meta, followed with a &#8220;PyTorch Updates&#8221; keynote, giving the community a look at the latest developments in the core framework.</span></p>
<p><b>Joe Spisak</b><span style="font-weight: 400;">, VP of Product and Head of Open Source at Reflection AI, took the stage next with &#8220;Community Led Open Source RL,&#8221; highlighting how the reinforcement learning community is pushing boundaries through open collaboration.</span></p>
<p><b>Ramine Roane</b><span style="font-weight: 400;">, Corporate Vice President of AI Product Management and Ecosystem Development at AMD, delivered a keynote titled &#8220;From One Node to Distributed Training and Inference: How the PyTorch Ecosystem Changed AI,&#8221; tracing the journey from single-node experiments to planet-scale deployments.</span></p>
<p><b>Patrick von Platen</b><span style="font-weight: 400;">, Research Engineer at Mistral AI, presented &#8220;Stream Everything: Moving from Request Input to Streaming Input,&#8221; offering a forward-looking vision of how AI systems will process continuous streams of data rather than discrete requests.</span></p>
<p><b>Maryam Tahhan</b><span style="font-weight: 400;">, Principal Engineer, Red Hat and </span><b>Nicolo Lucchesi</b><span style="font-weight: 400;">, Senior Machine Learning Engineer, Red Hat delivered a keynote, &#8220;Any [Agent | Model | Accelerator | Cloud]: Open Source AI Unlocks the World&#8217;s Potential,&#8221; making the case that open source is the path to truly flexible, portable AI.</span></p>
<p><span style="font-weight: 400;"><img alt="" class="aligncenter wp-image-66035 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55192994621_b5b3cd1f7f_o.jpg" width="2400" /></span></p>
<p><span style="font-weight: 400;">The morning keynote block closed with </span><b>Besmira Nushi</b><span style="font-weight: 400;">, Senior Manager of AI Research at NVIDIA, presenting &#8220;The Unbearable Lightness of (Agentic) Evaluations,&#8221; a thought-provoking look at the challenges of evaluating agentic AI systems.</span></p>
<p><b>Meet the Developers</b></p>
<p><span style="font-weight: 400;">A highlight of the conference format was the &#8220;Meet the Developers&#8221; sessions, where attendees could sit down face to face with the people behind the projects. On Day 1, </span><b>Meet the Developers of PyTorch Module Maintainers </b><span style="font-weight: 400;"> featured Driss Guessous, Mergen Nachin, Natalia Gimelshein, Jason Ansel, Edward Yang, and Alban Desmaison. Later in the afternoon, a dedicated </span><b>Meet the Developers of Helion</b><span style="font-weight: 400;"> brought together Jason Ansel, Oguz Ulgen, Will Feng, and Markus Hoehnerbach.</span></p>
<p><b><img alt="" class="aligncenter wp-image-66150 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55205736594_8555706a73_o.jpg" width="2400" /></b></p>
<p><b>Frameworks and Compilers</b></p>
<p><span style="font-weight: 400;">This track was one of the most densely packed at the conference. Standout sessions included:</span></p>
<ul>
<li style="font-weight: 400;"><b>Helion 1.0: A High-Level DSL for Performance Portable Kernels</b><span style="font-weight: 400;"> by Oguz Ulgen from Meta, introducing the newly contributed Helion project as a first-class tool for kernel authoring in PyTorch.</span></li>
<li style="font-weight: 400;"><b>Parameterized CUDA Graph Launch in PyTorch: CUDA Graphs Without the Pain</b><span style="font-weight: 400;"> by Daniel Galvez from NVIDIA, tackling one of the trickiest performance optimization challenges.</span></li>
<li style="font-weight: 400;"><b>FlexAttention + FlashAttention-4: Fast and Flexible</b><span style="font-weight: 400;"> by Driss Guessous from Meta, showcasing the next generation of attention mechanisms.</span></li>
<li style="font-weight: 400;"><b>Combo Kernels: Horizontal Fusion Optimization in Torch.compile</b><span style="font-weight: 400;"> by Karthick Panner Selvam and Elias Ellison from Meta.</span></li>
<li style="font-weight: 400;"><b>Model-Changing Transforms with Torch.compile</b><span style="font-weight: 400;"> by Thomas Viehmann from Lightning AI.</span></li>
<li style="font-weight: 400;"><b>Brevitas Quantization Library</b><span style="font-weight: 400;"> by Pablo Monteagudo Lago from AMD.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66034 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55192994921_45848c42dc_o.jpg" width="2400" /></b></p>
<p><b>Inference and Production</b></p>
<p><span style="font-weight: 400;">The inference track reflected the community&#8217;s growing focus on getting models into production efficiently:</span></p>
<ul>
<li style="font-weight: 400;"><b>Tour De Force: LLM Inference Optimization From Simple to Sophisticated</b><span style="font-weight: 400;"> by Christin Pohl from Microsoft, a comprehensive walkthrough on the Master Stage.</span></li>
<li style="font-weight: 400;"><b>ExecuTorch on Microcontrollers: Deploying PyTorch to the Smallest Edge</b><span style="font-weight: 400;"> by RJ Ascani and Matthias Cremon from Meta, pushing the boundaries of where PyTorch can run.</span></li>
<li style="font-weight: 400;"><b>Write Once, Run Everywhere with PyTorch Transformers</b><span style="font-weight: 400;"> by Pedro Cuenca from Hugging Face.</span></li>
<li style="font-weight: 400;"><b>Why WideEP Inference Needs Data-Parallel-Aware Scheduling</b><span style="font-weight: 400;"> by Maroon Ayoub (IBM) and Tyler Michael Smith (Red Hat).</span></li>
<li style="font-weight: 400;"><b>The Token Slice: Implementing Preemptive Scheduling via Chunked Decoding</b><span style="font-weight: 400;"> by Maroon Ayoub (IBM) and Kellen Swain (Google).</span></li>
<li style="font-weight: 400;"><b>Cross-Region Model Serving: PyTorch Inference, Observability, and LLMOps</b><span style="font-weight: 400;"> by Suraj Muraleedharan from Amazon Web Services.</span></li>
<li style="font-weight: 400;"><b>Accelerating On-Device ML Inference with ExecuTorch and Arm SME2</b><span style="font-weight: 400;"> by Jason Zhu from Arm.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66067 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55205480286_3bceb6084b_o.jpg" width="2400" /></b></p>
<p><b>GenAI and Multimodal</b></p>
<ul>
<li style="font-weight: 400;"><b>Lights, Camera, Inference! Video Generation as a Service with VLLM-Omni</b><span style="font-weight: 400;"> by Ricardo Noriega and Doug Smith from Red Hat.</span></li>
<li style="font-weight: 400;"><b>The Science and Practice of Open and Scalable LLM Evaluations</b><span style="font-weight: 400;"> by Grzegorz Chlebus from NVIDIA.</span></li>
<li style="font-weight: 400;"><b>torch.compile and Diffusers: A Hands-On Guide to Peak Performance</b><span style="font-weight: 400;"> by Sayak Paul from Hugging Face.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66020 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55205480106_687f1cc2b6_o.jpg" width="2400" />Training Systems</b></p>
<ul>
<li style="font-weight: 400;"><b>Training Embedding Model Resiliently for Multimodal Model Inference Routing</b><span style="font-weight: 400;"> by Huamin Chen (Red Hat) and Haichen Zhang (AMD).</span></li>
<li style="font-weight: 400;"><b>TorchJD: Jacobian Descent in PyTorch</b><span style="font-weight: 400;"> by Pierre Quinton (EPFL) and Valerian Rey (Simplex Lab).</span></li>
<li style="font-weight: 400;"><b>Bringing Google&#8217;s Colossus to PyTorch: Rapid Storage via fsspec to Keep GPUs Busy</b><span style="font-weight: 400;"> by Ankita Luthra and Trinadh Kotturu from Google.</span></li>
<li style="font-weight: 400;"><b>Jigsaw: Domain and Tensor Parallelism for High-Resolution Input Training</b><span style="font-weight: 400;"> by Deifilia Kieckhefen from Karlsruhe Institute of Technology.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66147 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55194161162_8bb818160b_o.jpg" width="2400" /></b></p>
<p><b>Applications and Case Studies</b></p>
<ul>
<li style="font-weight: 400;"><b>Deep Learning in the Wild: Embedded PyTorch for Real-World Conservation Bioacoustics</b><span style="font-weight: 400;"> by Taraqur Rahman and Owen O&#8217;Donnell from OWL Integrations.</span></li>
<li style="font-weight: 400;"><b>How DeepInverse Is Solving Imaging in Science and Healthcare with PyTorch</b><span style="font-weight: 400;"> by Andrew Wang (DeepInverse) and Minh Hai Nguyen (Universite de Toulouse).</span></li>
<li style="font-weight: 400;"><b>Flexible Deployment of PyTorch Models on MCU-Class Devices Using ExecuTorch</b><span style="font-weight: 400;"> by Robert Kalmar and Martin Pavella from NXP.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66151 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55204580822_57a99e26f5_o.jpg" width="2400" /></b></p>
<p><b>Responsible AI, Security, and Privacy</b></p>
<ul>
<li style="font-weight: 400;"><b>Engineering for the EU AI Act: What Should PyTorch Expose Natively?</b><span style="font-weight: 400;"> &#8211; a Birds of a Feather session led by Roy Saurabh from AffectLog, exploring the intersection of regulation and framework design.</span></li>
<li style="font-weight: 400;"><b>From Pretrained to Personal: Privacy-First Fine-Tuning on AI PCs</b><span style="font-weight: 400;"> by Daniel Holanda Noronha and Iswarya Alex from AMD.</span></li>
<li style="font-weight: 400;"><b>Ethical, Privacy and Sustainability Considerations in PyTorch Systems</b><span style="font-weight: 400;"> by Paula Mesa Macias from Pau&amp;Company.</span></li>
<li style="font-weight: 400;"><b>Why Classic IAM Collapses for Agents: Rethinking IAM for Agentic Systems</b><span style="font-weight: 400;"> by Parul Singh from Red Hat.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66143 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55192115332_80df12fcde_o.jpg" width="2400" /></b></p>
<p><b>Community Events</b></p>
<p><span style="font-weight: 400;">Day 1 also featured a </span><b>Women and Non-Binary in PyTorch Lunch</b><span style="font-weight: 400;"> and a lively </span><b>Flare Party</b><span style="font-weight: 400;"> in the evening featuring good food, good drinks, and good company. Both days featured poster presentations across every track, giving researchers and engineers the chance to share work in progress and engage in one-on-one technical conversations. Poster tracks covered Applications and Case Studies, Frameworks and Compilers, GenAI and Multimodal, Inference and Production, and Responsible AI and Compliance.</span></p>
<p><b><img alt="" class="aligncenter wp-image-66021 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55205624668_dfe22d76a7_o-1.jpg" width="2400" /></b></p>
<p><b>Day 2: Wednesday, April 8</b></p>
<p><b>Morning Keynotes</b></p>
<p><span style="font-weight: 400;">Day 2 opened with another strong keynote lineup on the Master Stage.</span></p>
<p><b>Matt White</b><span style="font-weight: 400;">, PyTorch CTO and Global CTO of AI at the Linux Foundation, set the stage for the day.</span></p>
<p><b>Tyler Michael Smith</b><span style="font-weight: 400;">, Chief Architect, Inference Engineering, Red Hat and </span><b>Artur Niederfahrenhorst</b><span style="font-weight: 400;">, Member of Technical Staff, Anyscale delivered a joint keynote talk on </span><b>vLLM and Ray Updates</b><span style="font-weight: 400;">, two projects that have become cornerstones of production AI infrastructure.</span></p>
<p><b>Lysandre Debut</b><span style="font-weight: 400;">, Chief Open-Source Officer at Hugging Face, presented &#8220;The Hub as Infrastructure: From Open PyTorch Models, to a Safe and Performant Distribution Hub,&#8221; underscoring how the Hugging Face Hub has evolved from a model repository into critical AI infrastructure.</span></p>
<p><b>Jonathan Bryce</b><span style="font-weight: 400;">, Executive Director of the Cloud Native Computing Foundation (CNCF), gave a keynote on &#8220;Open Source Infrastructure for the AI Native Era,&#8221; connecting the dots between cloud-native and AI-native development.</span></p>
<p><b>Leonard Hussenot</b><span style="font-weight: 400;">, Research Scientist at Google DeepMind, closed the keynote block with &#8220;Gemma 4: Compacting Intelligence for the Edge,&#8221; a look at how Google DeepMind is making powerful models small enough to run on edge devices.</span></p>
<p><b><img alt="" class="aligncenter wp-image-66093 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55195302914_8d96f63304_o-1.jpg" width="2400" />Frameworks and Compilers (Day 2)</b></p>
<ul>
<li style="font-weight: 400;"><b>Monarch: An API to Your Supercomputer</b><span style="font-weight: 400;"> by Marius Eriksen from Meta.</span></li>
<li style="font-weight: 400;"><b>How to Write C++ Extensions in 2026</b><span style="font-weight: 400;"> by Jane Xu and Mikayla Gawarecki from Meta.</span></li>
<li style="font-weight: 400;"><b>Achieving SOTA GEMM Performance: A CuTeDSL Backend for PyTorch Inductor</b><span style="font-weight: 400;"> by Nikhil Patel from Meta.</span></li>
<li style="font-weight: 400;"><b>Accelerating PyTorch Models with Torch.compile&#8217;s C++ Wrapper Mode</b><span style="font-weight: 400;"> by Bin Bao from Meta.</span></li>
<li style="font-weight: 400;"><b>Bringing PyTorch Monarch to AMD GPUs: Single-Controller Distributed Training on ROCm</b><span style="font-weight: 400;"> by Liz Li and Zachary Streeter from AMD.</span></li>
<li style="font-weight: 400;"><b>PyTorch on RISC-V: From Cross-Compilation to Native CI</b><span style="font-weight: 400;"> by Ludovic Henry from Meta.</span></li>
<li style="font-weight: 400;"><b>PyTorch Symmetric Memory + NCCL Device APIs: A New Path Towards Multi-GPU Kernels</b><span style="font-weight: 400;"> by Ke Wen and Sylvain Jeaugey from NVIDIA.</span></li>
<li style="font-weight: 400;"><b>Seamless Integration: Custom Kernels in the Torch.compile Stack Without Graphbreaks</b><span style="font-weight: 400;"> by Kshiteej Kalambarkar, Masaki Kozuki, and Pawel Gadzinski from NVIDIA.</span></li>
<li style="font-weight: 400;"><b>Faster Than SOTA Kernels in Torch.compile with Subgraph Fusions and Custom Op Autotuning</b><span style="font-weight: 400;"> by Elias Ellison and Paul Zhang from Meta.</span></li>
<li style="font-weight: 400;"><b>De-mystifying PyTorch for ASICs: When (and Why) to Move Your Development to AI Accelerators</b><span style="font-weight: 400;"> by Alpha Romer Coma from Kollab Philippines.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66116 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55204526627_ccd1e95d96_o.jpg" width="2400" /></b></p>
<p><b>Inference and Production (Day 2)</b></p>
<ul>
<li style="font-weight: 400;"><b>KV-Cache Centric Inference: Building a State-Aware Serving Platform with Llm-d and VLLM</b><span style="font-weight: 400;"> by Martin Hickey and Maroon Ayoub from IBM Research.</span></li>
<li style="font-weight: 400;"><b>Optimizing Large MoE Inference on NVIDIA Blackwell: NVFP4, ADP, and DualPipe Strategies</b><span style="font-weight: 400;"> by Julien Demouth from NVIDIA.</span></li>
<li style="font-weight: 400;"><b>Portable High-Performance LLM Serving: A Triton Backend for VLLM</b><span style="font-weight: 400;"> by Burkhard Ringlein and Jan van Lunteren from IBM.</span></li>
<li style="font-weight: 400;"><b>Deploying PyTorch Models to the Browser and Beyond with Transformers.js</b><span style="font-weight: 400;"> by Joshua Lochner from Hugging Face.</span></li>
<li style="font-weight: 400;"><b>Inside VLLM&#8217;s KV Offloading Connector: Async Memory Transfers for Higher Inference Throughput</b><span style="font-weight: 400;"> by Nicolo Lucchesi from Red Hat.</span></li>
<li style="font-weight: 400;"><b>Optimizing CPU LLM Inference in PyTorch: Lessons from VLLM</b><span style="font-weight: 400;"> by Crefeda Rodrigues and Fadi Arafeh from Arm.</span></li>
<li style="font-weight: 400;"><b>From Hugging Face to Handheld: Scaling LLM Deployment with LiteRT Generative API</b><span style="font-weight: 400;"> by Cormac Brick and Weiyi Wang from Google.</span></li>
<li style="font-weight: 400;"><b>Full-Stack PyTorch Robotics VLA: From Data to Edge via ExecuTorch/OpenVINO</b><span style="font-weight: 400;"> by Samet Akcay and Dmitriy Pastushenkov from Intel.</span></li>
<li style="font-weight: 400;"><b>Beyond the Theory: What Actually Breaks When You Scale Your Disaggregated PyTorch Models</b><span style="font-weight: 400;"> by Ekin Karabulut and Ron Kahn from NVIDIA.</span></li>
<li style="font-weight: 400;"><b>Slash LLM Cold-Start Times by Pre-distributing GPU Caches</b><span style="font-weight: 400;"> by Billy McFall and Maryam Tahhan from Red Hat.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66092 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55195458285_9e6e7fa60c_o.jpg" width="2400" /></b></p>
<p><b>Training Systems (Day 2)</b></p>
<ul>
<li style="font-weight: 400;"><b>FP8 Training from Hopper to Blackwell</b><span style="font-weight: 400;"> by Luca Wehrstedt from Meta.</span></li>
<li style="font-weight: 400;"><b>Backpropagation-Free Optimization in PyTorch</b><span style="font-weight: 400;"> by Andrii Krutsylo from the Polish Academy of Sciences.</span></li>
<li style="font-weight: 400;"><b>Debugging the Undebuggable: Introducing Torch.distributed.debug</b><span style="font-weight: 400;"> by Tristan Rice from Meta.</span></li>
<li style="font-weight: 400;"><b>Scaling Recommendation Systems to 2K GPUs and Beyond</b><span style="font-weight: 400;"> by Zain Huda from Meta.</span></li>
<li style="font-weight: 400;"><b>From Responses to Trajectories: Multi-Turn and Multi-Environment Reinforcement Learning</b><span style="font-weight: 400;"> by Kashif Rasul and Sergio Paniego Blanco from Hugging Face.</span></li>
<li style="font-weight: 400;"><b>Trinity Large: Torchtitan on 2000+ B300s</b><span style="font-weight: 400;"> by Matej Sirovatka from Prime Intellect.</span></li>
<li style="font-weight: 400;"><b>DualPipe from Scratch: Implementing DeepSeek&#8217;s 5D Parallelism in PyTorch</b><span style="font-weight: 400;"> by Dev Jadhav from ING Bank.</span></li>
<li style="font-weight: 400;"><b>Fault-Tolerant Training: How We Build Reliable Clusters for Distributed AI Workloads</b><span style="font-weight: 400;"> by Cyril Konkratenko and Maurits de Groot from Nebius.</span></li>
<li style="font-weight: 400;"><b>Why Logging Isn&#8217;t Enough: Making PyTorch Training Regressions Visible in Practice</b><span style="font-weight: 400;"> by Sahana Venkatesh from Wayve.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66094 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55193147813_798268156b_o.jpg" width="2400" /></b></p>
<p><b>Agents and Interop</b></p>
<ul>
<li style="font-weight: 400;"><b>Beyond JSON-RPC: Scaling Model Context Protocols with gRPC in the PyTorch Ecosystem</b><span style="font-weight: 400;"> by Ashesh Vidyut and Madhav Bissa from Google.</span></li>
<li style="font-weight: 400;"><b>Coding Agents for Compiler Construction: Beyond the AI Assistant Paradigm</b><span style="font-weight: 400;"> by Reza Rahimi from yasp.ai and Stefan Krassin from yasp.ai.</span></li>
<li style="font-weight: 400;"><b>Bridging the Hardware Gap with Code Harnesses on the Hugging Face Kernels Hub</b><span style="font-weight: 400;"> by Ben Burtenshaw from Hugging Face.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66103 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55205665704_809a395ed2_o.jpg" width="2400" /></b></p>
<p><b>Security and Privacy</b></p>
<ul>
<li style="font-weight: 400;"><b>Live Migration of PyTorch GPU Nodes from Azure to European Clouds</b><span style="font-weight: 400;"> by Mike Krom from Acf Cyber Solutions.</span></li>
<li style="font-weight: 400;"><b>Securing Agentic AI with PyTorch: Threat Modeling and LLM Red Teaming in Practice</b><span style="font-weight: 400;"> by Valeri Milke from VamiSec GmbH.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66146 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55194160412_69980f2047_o-1.jpg" width="2400" /></b></p>
<p><b>Responsible AI and Compliance</b></p>
<ul>
<li style="font-weight: 400;"><b>From Gradients to Governance: Making PyTorch Lineage-Aware</b><span style="font-weight: 400;"> by Kateryna Romashko and Clodagh Walsh from Red Hat.</span></li>
<li style="font-weight: 400;"><b>Building Trust for Users and Regulators Alike: A Cost-Efficient PyTorch Path to Compliance-as-Code</b><span style="font-weight: 400;"> by Raja Gopal Hari Vijay from Zoho Corporation.</span></li>
<li style="font-weight: 400;"><b>Bridging the Gap: Engineering Compliant &#8220;Glass Box&#8221; Medical AI with PyTorch</b><span style="font-weight: 400;"> by Muhammad Saqib Hussain and Mohaddisa Maryam from Neurosonic.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66117 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55204528617_8ec7f0d774_o.jpg" width="2400" /></b></p>
<p><b>Applications and Case Studies (Day 2)</b></p>
<ul>
<li style="font-weight: 400;"><b>Ball Tracking and Detection in Soccer Videos: Comparison of VLMs and Traditional Pipelines</b><span style="font-weight: 400;"> by Maciej Szymkowski from Future Processing.</span></li>
</ul>
<p><b><img alt="" class="aligncenter wp-image-66119 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55204522252_2f7493c345_o.jpg" width="2400" /></b></p>
<p><b>Birds of a Feather and Meet the Maintainers</b></p>
<p><span style="font-weight: 400;">Day 2 featured two compelling Birds of a Feather sessions: </span><b>Disaggregated Tokenization: Building Toward Tokens-In-Tokens-Out LLM Inference</b><span style="font-weight: 400;"> with Maroon Ayoub, IBM Research; Hang Yin &amp; Xi Ning Wang, Alibaba Cloud; Nili Guy, IBM; and Hyunkyun Moon, Moreh and </span><b>NCCL in the Wild: Scaling Communications to Thousands of GPUs</b><span style="font-weight: 400;"> with Jeff Hammond, Gabrielle Talavera, Ke Wen, and Asma Farjallah, NVIDIA.</span></p>
<p><span style="font-weight: 400;"><img alt="" class="aligncenter wp-image-66118 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55204520592_0761d44f81_o.jpg" width="2400" /></span></p>
<p><span style="font-weight: 400;">Attendees also had the opportunity to </span><b>Meet the vLLM Maintainers</b><span style="font-weight: 400;"> (Tyler Michael Smith and Nicolo Lucchesi) and </span><b>Meet the Ray Maintainers</b><span style="font-weight: 400;"> (Artur Niederfahrenhorst).</span></p>
<p><b><img alt="" class="aligncenter wp-image-66016 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55193148108_d34190a029_o-1.jpg" width="2400" /></b></p>
<p><b>Community Expo and Networking</b></p>
<p><span style="font-weight: 400;">The Community Expo ran throughout both days, giving attendees the chance to explore demos, connect with project maintainers, and discover new tools in the ecosystem. Sponsor activities included a session on &#8220;Validating AI on CPUs: The vLLM 3-Phase Evaluation Framework&#8221; and &#8220;Lobster Trap: OpenClaw in Containers.&#8221;</span></p>
<p><b><img alt="" class="aligncenter wp-image-66124 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55205813725_37c9f1aa5a_o.jpg" width="2400" /></b></p>
<p><b>Key Themes and Takeaways</b></p>
<p><b>The open source AI stack is maturing.</b><span style="font-weight: 400;"> With Helion and Safetensors joining as hosted-projects and ExecuTorch becoming part of PyTorch Core, the PyTorch ecosystem now covers an even wider range of the AI lifecycle. From secure model distribution to edge deployment to high-performance kernel authoring, these projects fill critical gaps.</span></p>
<p><b>Inference is the new frontier.</b><span style="font-weight: 400;"> A large share of sessions focused on production deployment, from vLLM and LLM serving optimizations to ExecuTorch on microcontrollers and in-browser inference with Transformers.js. The community is clearly moving beyond training and into making models work in the real world.</span></p>
<p><b>Agentic AI demands new infrastructure.</b><span style="font-weight: 400;"> Several sessions tackled the unique challenges of agentic systems, including rethinking Identity and Access Management (IAM), building trust through evaluation, and securing agentic pipelines. This is a space to watch.</span></p>
<p><b>Europe is stepping up.</b><span style="font-weight: 400;"> From EU AI Act compliance sessions to talks on data sovereignty and cloud migration, the conference reflected Europe&#8217;s growing role in shaping responsible AI development. The inaugural European conference was a milestone for the global community.</span></p>
<p><b>Hardware diversity is expanding.</b><span style="font-weight: 400;"> Sessions covered AMD ROCm, Arm SME2, Google TPUs, NVIDIA Blackwell, RISC-V, microcontrollers, Qualcomm QNN, Intel OpenVINO, and more. PyTorch&#8217;s hardware portability story has never been stronger.</span></p>
<p><b><img alt="" class="aligncenter wp-image-66018 size-full" height="1600" src="https://pytorch.org/wp-content/uploads/2026/04/55195048196_b4b917af6c_o.jpg" width="2400" /></b></p>
<p><b>Thank You, Paris</b></p>
<p><span style="font-weight: 400;">PyTorch Conference Europe 2026 was a landmark event for the community. To every sponsor, speaker, poster presenter, and attendee who joined us in Paris: thank you. The energy, the technical depth, and the spirit of open collaboration were truly remarkable. Hats of to the PyTorchCon Europe Program Chairs: Alban Desmaison, Lysandre Debut, and Luca Antiga and the full Program Committee for putting together a successful event!</span></p>
<p><span style="font-weight: 400;">All session recordings will be available on the</span><a href="https://www.youtube.com/@PyTorch"> <span style="font-weight: 400;">PyTorch YouTube Channel</span></a><span style="font-weight: 400;">. Browse the full photo album from the event on</span><a href="https://www.flickr.com/photos/197037482@N07/albums/72177720332470766/"> <span style="font-weight: 400;">Flickr</span></a><span style="font-weight: 400;">.</span></p>
<p><span style="font-weight: 400;">See you at our next PyTorch Conferences including</span><a href="https://www.lfopensource.cn/kubecon-cloudnativecon-openinfra-summit-pytorch-conference-china/"> <span style="font-weight: 400;">PyTorch Conference China</span></a><span style="font-weight: 400;"> – September 8-9 in Shanghai, China and</span><a href="https://events.linuxfoundation.org/pytorch-conference-north-america/"> <span style="font-weight: 400;">PyTorch Conference North America</span></a><span style="font-weight: 400;"> – October 20-21 in San Jose, CA later this year.</span></p>
