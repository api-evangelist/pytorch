---
title: "Monarch: an API to your supercomputer"
url: "https://pytorch.org/blog/monarch-an-api-to-your-supercomputer/"
date: "Wed, 08 Apr 2026 07:00:27 +0000"
author: "The PyTorch Team at Meta"
feed_url: "https://pytorch.org/feed/"
---
<p>Getting distributed training jobs to run on huge clusters is hard!  This is especially true when you start looking at more complex setups like distributed reinforcement learning. Debugging these kinds of jobs is frustrating, and the turnaround time for changes tends to be very slow.</p>
<p>Monarch is a distributed programming framework for PyTorch that makes the cluster programmable through a simple Python API. It exposes the supercomputer as a coherent, directly controllable system—bringing the experience of local development to large-scale training, as if your laptop had 1000s of GPUs attached.  A complete training system can be defined in a single Python program. Core primitives are explicit and minimal, enabling higher-level capabilities—fault tolerance, orchestration, tooling integration—to be built as reusable libraries.</p>
<p>Monarch is optimized for agentic usage, providing consistent infrastructure abstractions and exposing telemetry via standard SQL-based APIs that agents already excel at using. Agents can do a lot of development tasks by just running on your dev machine, and Monarch is really good at turning your devmachine into a supercomputer, leveling-up those agents.</p>
<p>The project launched at the PyTorch conference in October 2025; you can read about it here: <a href="https://pytorch.org/blog/introducing-pytorch-monarch/">Introducing PyTorch Monarch</a>. This blog covers how Monarch has evolved into an effective framework for agent-driven training development.  It will also cover Monarch’s major improvements since October, including native Kubernetes support, RDMA improvements, distributed telemetry, and more.</p>
<h2>Agentic Development in Monarch</h2>
<p>By representing your supercomputing cluster through a coherent model of hosts, procs, and actors, and pairing it with “batteries included” infrastructure, Monarch gives your agent superpowers! It can directly manage and debug running code, rapidly sync dependencies and data, run new code, and provision additional hosts, procs, and actors in an efficient and consistent way regardless of where it is deployed.</p>
<p>Let’s quickly review some key features Monarch uses to empower agentic development:</p>
<ul>
<li>RDMA-Powered Remote File System &#8211; Distribute files from the client on a read-only mounted filesystem to every host in the job via RDMA.  This lets you very rapidly sync code, dependencies, and containers while iterating on the machine learning ideas. Monarch&#8217;s RDMA filesystem in turn is built on Monarch RDMA buffers and PyFuse.</li>
<li>Distributed SQL Telemetry &#8211; Use Monarch’s integrated lightweight distributed SQL engine to collect live state information, pyspy traces, and logs from all distributed processes/actors/etc. We used Monarch to directly run a DataFusion distributed SQL query engine *in situ*; each node in turn writes live state information into a set of tables that can then be queried directly and efficiently by an agent.  This makes it very easy to explore the state of the system when debugging.</li>
<li>Jobs API &#8211; Provision resources (hosts) once and run as many jobs as needed on them without paying the repeated allocation penalty. Monarch comes with support for Kubernetes and SLURM; other schedulers can be integrated by implementing a Monarch Job.</li>
</ul>
<p>Collectively, these features enable agents to be efficient across some key phases of development; they can restart jobs fast, sync new code, dependencies, and data fast, and debug fast, all from a central point.  In short, Monarch makes the distributed system feel local and provides a toolbox to reduce the iteration time when tackling problems.</p>
<h2>What’s new in Monarch?</h2>
<p>Let’s review what is new in Monarch since its launch at the PyTorch Conference in October 2025 (~6 months ago).</p>
<h3>Kubernetes</h3>
<p>Monarch now has first-class Kubernetes support.</p>
<ul>
<li>Monarch-kubernetes OSS repository &#8211; A dedicated repo (<a href="https://github.com/meta-pytorch/monarch-kubernetes">github.com/meta-pytorch/monarch-kubernetes</a>) with a MonarchMesh Custom Resource Definition, a reference KubeBuilder operator, and a hello-world demo.  The MonarchMesh label propagation also enables scheduling via Kueue.</li>
<li>Just-in-time pod provisioning &#8211; Pods are allocated on demand rather than reserved upfront, improving cluster utilization.</li>
<li>External gateway &#8211; Out-of-cluster clients can now connect to Monarch meshes running inside Kubernetes (landing in 0.5).</li>
<li>Versioned and nightly Docker containers &#8211; Published to <a href="https://github.com/meta-pytorch/monarch/pkgs/container/monarch">GHCR</a> for reproducible deployments.</li>
</ul>
<h3>RDMA &amp; Networking</h3>
<p>Monarch has continued its investment in RDMA, adding support for multiple new backends and providing a higher-level API to make supporting and using them easier.</p>
<ul>
<li>AWS EFA RDMA support &#8211; Monarch&#8217;s RDMABuffer now supports Elastic Fabric Adapter (EFA) on AWS, extending high-performance networking beyond InfiniBand. Validated at 16 Gbps &#8211; 10x faster than TCP (14.5 GB in 7.6 seconds). Available in PyPI nightlies.</li>
<li>AMD ROCm GPU support &#8211; GPU-direct RDMA and RCCL collective communication now work on AMD GPUs via ROCm with Mellanox interfaces.</li>
<li>Unified RDMA API &#8211; A hardware-portable RDMA interface that works across InfiniBand (mlx5), AWS EFA, and ROCm, letting users write once and run on any fabric, or fall back to Monarch actor messaging when not available.</li>
</ul>
<h3>Observability &amp; Telemetry</h3>
<p>Monarch has leaned heavily into observability &amp; telemetry, adding programmatic mechanisms to empower agentic development.  There are also new native dashboards, Terminal UI (TUIs), and support for OSS standards commonly used by DevOps teams.</p>
<ul>
<li>Distributed SQL Telemetry &#8211; A client-accessible SQL endpoint, enabling easy analysis of the distributed system without 3rd party dependencies.</li>
<li>Admin API &amp; Terminal UI &#8211; A terminal-based interface for inspecting and managing live Monarch jobs, backed by a powerful API for accessing internals.<img alt="" class="aligncenter wp-image-62240" height="734" src="https://pytorch.org/wp-content/uploads/2026/04/pytblog_040626a.png" width="1212" /></li>
<li style="font-weight: 400;"><b>OpenTelemetry integration</b><span style="font-weight: 400;"> &#8211; Native OTel support for metrics, logs, and visualization on Kubernetes, giving users full observability on any cluster.  This is easily integrated with Prometheus, Loki, Grafana, and other common OSS tooling.</span></li>
<li style="font-weight: 400;"><b>Per-job OSS dashboard (Beta)</b> &#8211; A built-in web dashboard for visualizing and debugging distributed jobs without external tooling.</li>
</ul>
<h3><img alt="" class="aligncenter wp-image-62243 size-full" height="658" src="https://pytorch.org/wp-content/uploads/2026/04/pytblog_040626b.png" width="1183" /></h3>
<h3>Portability &amp; Installation</h3>
<p>Monarch is now significantly more compact and much faster to start, making it easier to use than ever.</p>
<ul>
<li>100x smaller install, 8x faster startup &#8211; The pip wheel footprint was reduced by two orders of magnitude with dramatically faster cold-start times. libpython linking requirements were removed entirely.</li>
<li>Torch dependency removed &#8211; As of v0.2, torchmonarch no longer pulls in torch as a pip dependency, simplifying installation and avoiding version conflicts.</li>
<li>Native uv support &#8211; Monarch works out of the box with <a href="https://github.com/astral-sh/uv">uv</a>, the fast Python package manager. Three commands to get started: git clone, cd, uv run example.py. See the <a href="https://github.com/allenwang28/monarch-uv">example repo</a>.</li>
<li>Consolidated PyPI packaging &#8211; All packages unified under a single torchmonarch name with PEP 440 pre-release versioning for nightlies: pip install &#8211;pre torchmonarch. ARM64 Linux builds are added as well to v0.4</li>
</ul>
<h3>Developer Experience</h3>
<ul>
<li>Interactive SPMD &#8211; Improved support for interactive, notebook-style development with SPMD (Single Program, Multiple Data) jobs.</li>
<li>RDMA File System &#8211; Fast, convenient file-sync across hosts.</li>
</ul>
<h2>Collaborations</h2>
<p>We’d also like to take a moment to acknowledge some collaborators that have helped make Monarch better since its release.</p>
<ul>
<li><a href="https://docs.skypilot.co/en/stable/docs/index.html"><strong>SkyPilot</strong></a>
<ul>
<li>Run Monarch on any Kubernetes cluster with a single command &#8211; the SkyPilot integration lets users sky launch Monarch workloads on any K8s cluster or cloud without changing their Monarch code. Great for teams that need GPUs wherever they&#8217;re available.</li>
<li>Multi-node distributed training with zero infra setup &#8211; SkyPilot handles node provisioning, networking, and gang scheduling so users can focus on their Monarch training logic. The integration uses Monarch&#8217;s JobTraits API to plug into SkyPilot as the job backend. No need to install separate operators on your k8s clusters.</li>
<li>See  <a href="https://github.com/meta-pytorch/monarch/tree/main/examples/skypilot">https://github.com/meta-pytorch/monarch/tree/main/examples/skypilot</a> for more.</li>
</ul>
</li>
<li><strong><a href="https://github.com/volcengine/verl">VERL</a></strong>
<ul>
<li>VERL is a popular open-source framework for distributed RLHF post-training. In collaboration with ByteDance&#8217;s VeRL team, we developed a Monarch backend for VeRL&#8217;s single-controller architecture, implementing new resource pool abstractions built on Monarch&#8217;s Job API, colocated multi-role worker support, an RDMA-based transport layer that moves tensors out-of-band for VeRL&#8217;s DataProto exchange pattern, and a vLLM server integration that solves actor handle discovery without relying on a global actor registry. We validated that VeRL&#8217;s PPO and GRPO training loops can run on Monarch through this backend using VeRL&#8217;s hybrid-engine training mode, producing numerically identical results with no performance regression. One finding from this work: while VeRL&#8217;s single-controller interface is cleanly abstracted, Ray API usage surfaces throughout the broader codebase — making a non-invasive backend swap more involved than the interface alone suggests. This is a common pattern in frameworks built on Ray, and something the Monarch and VeRL communities can collaborate on over time.</li>
</ul>
</li>
<li><strong>AMD</strong>
<ul>
<li>Monarch expanded its compatibility and performance across leading hardware infrastructure adding AMD as a supported platform. Our partners at AMD validated Monarch on their ROCm platform, enabling seamless SLURM-based orchestration for MI300/325/355 clusters. This integration allows users to efficiently schedule, manage, and scale AI workloads across AMD GPUs, leveraging the familiar SLURM ecosystem widely used in HPC and AI research.</li>
<li>Thanks to their effort, Monarch now supports RDMA (Remote Direct Memory Access) for fast GPU-to-GPU communication on AMD clusters equipped with Mellanox network interfaces. This hardware combination is available on major cloud providers like Azure and Oracle, enabling high-throughput, low-latency data transfers essential for distributed RL training and large-scale AI workloads.</li>
</ul>
</li>
</ul>
<h2>Conclusion</h2>
<p>Monarch is the API for your supercomputer; making distributed AI development feels like building a local app.  The future of AI training demands speed and simplicity Monarch provides for both humans and agents. We encourage you to explore the latest features, join our growing OSS-first development community, and help shape the next chapter of distributed computing.</p>
<h2>Acknowledgments</h2>
<p>Thank you to the whole Monarch team for making this work possible.  Also, a special thanks to our <a href="https://github.com/meta-pytorch/monarch/graphs/contributors">Top Contributors</a> on GitHub!</p>
<ul>
<li><img alt="🙏" class="wp-smiley" src="https://s.w.org/images/core/emoji/17.0.2/72x72/1f64f.png" style="height: 1em;" /> Special thanks to our partners at Google Cloud and Runhouse for helping integrate monarch with kubernetes, and to our partners at SkyPilot and AMD for their contributions!</li>
</ul>
<p>Ahmad Sharif, Allen Wang, Ali Sol, Amir Afzali, Carole-Jean Wu, Chris Gottbrath, Christian Puhrsch, Colin Taylor, Do Hyung (Dave) Kwon, Gayathri Aiyer, Hamid Shojanazeri, Jiyue Wang, Joe Spisak, John William Humphreys, Jun Li, Lucas Pasqualin, Marius Eriksen, Matthew Zhang, Matthias Reso, Peng Zhang, Riley Dulin, Rithesh Baradi, Robert Rusch, Sam Lurye, Samuel Hsia, Shayne Fletcher, Tao Lin, Thomas Wang, Victoria Dudin, Zachary DeVito</p>
