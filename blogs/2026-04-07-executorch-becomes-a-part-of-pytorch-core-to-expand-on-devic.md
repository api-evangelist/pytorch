---
title: "ExecuTorch Becomes a Part of PyTorch Core to Expand On-Device Inference Capabilities"
url: "https://pytorch.org/blog/executorch-becomes-part-of-pytorch-core/"
date: "Tue, 07 Apr 2026 07:10:19 +0000"
author: "PyTorch Foundation"
feed_url: "https://pytorch.org/feed/"
---
<p><span style="font-weight: 400;"><img alt="ExecuTorch Becomes a part of PyTorch Core" class="alignnone size-large wp-image-62343" height="576" src="https://pytorch.org/wp-content/uploads/2026/04/ExecuTorch-PyTorch-Core-1024x576.png" width="1024" /></span></p>
<p><span style="font-weight: 400;">Today, we’re excited to share that ExecuTorch is becoming a part of PyTorch Core. ExecuTorch extends PyTorch functionality for efficient AI inference on edge devices, from desktop/laptop to mobile phones and embedded systems. </span></p>
<p><span style="font-weight: 400;">Becoming a PyTorch Core project under the PyTorch Foundation will provide vendor‑neutral governance, clear IP, trademark, and branding, and ensure that business and ecosystem decisions are made transparently by a diverse group of members, while technical decisions remain with individual maintainers and open source contributors, ultimately strengthening ExecuTorch’s adoption within the PyTorch ecosystem.</span></p>
<p><span style="font-weight: 400;">At this moment, we want to reflect briefly on how ExecuTorch started, share why we’re becoming a PyTorch Core project, and what’s ahead.</span></p>
<h2><span style="font-weight: 400;">How ExecuTorch started</span></h2>
<p><span style="font-weight: 400;">ExecuTorch began at Meta as part of our effort to make it easier to run state-of-the-art PyTorch models efficiently on edge and on-device environments—from mobile phones and AR/VR headsets and Glasses to embedded devices and custom accelerators.</span></p>
<p><span style="font-weight: 400;">When we first introduced ExecuTorch publicly at PyTorch Conference 2023, it was designed around a small set of core principles:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">End-to-end developer experience: From authoring in PyTorch to deployment on-device, with a consistent, predictable workflow.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Portability across hardware: A runtime that could target a wide variety of CPUs, GPUs, NPUs, DSPs, and other accelerators across platforms.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Small, modular, and efficient: A lean runtime and a composable architecture suitable for constrained environments.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Open by default: A project positioned to benefit from and contribute to the broader open-source AI ecosystem.</span></li>
</ul>
<p><span style="font-weight: 400;">Since then, ExecuTorch has evolved from an internal runtime into an open platform for on-device AI. It underpins model deployment in Meta products and is increasingly being adopted by partners and the broader community as a flexible way to bring PyTorch-based models to production on edge devices.</span></p>
<h2><span style="font-weight: 400;">Growth and community</span></h2>
<p><span style="font-weight: 400;">ExecuTorch has quickly grown beyond its initial use cases. It is now used as a foundation for on-device inference across a variety of scenarios, including:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Mobile and AR/VR experiences</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Generative AI and LLM-based assistants on devices</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Computer vision and sensor processing at the edge</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Low-latency, privacy-preserving applications where models run locally</span></li>
</ul>
<p><span style="font-weight: 400;">While Meta has been the primary initial contributor to ExecuTorch, a growing set of companies and individual developers have started investing in the project—adding backends, operators, tooling, and integrations, as well as building their products and research efforts on top of ExecuTorch.</span></p>
<p><span style="font-weight: 400;">We see contributions and ecosystem work emerging around:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Hardware-specific optimizations and backends</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Tooling to convert, quantize, and package PyTorch models for ExecuTorch</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Integrations with mobile, AR/VR, IoT, and robotics platforms</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Benchmarks, testing, and documentation improvements</span></li>
</ul>
<p><span style="font-weight: 400;">ExecuTorch is becoming an important part of how organizations think about portable, hardware-agnostic on-device AI, and it’s clear the project is transitioning into a multi-stakeholder ecosystem. That makes this the right time to move to a broader open-source foundation.</span></p>
<h2><span style="font-weight: 400;">Why Become a PyTorch Core Project?</span></h2>
<p><span style="font-weight: 400;">In its early phase, ExecuTorch’s business governance was intentionally lightweight—we operated a lot like a small startup team within a larger organization. Meta helped put in the initial scaffolding: shaping the project’s roadmap, setting up basic contribution processes, aligning ExecuTorch with PyTorch’s model export and runtime stack, and engaging with early partners.</span></p>
<p><span style="font-weight: 400;">As ExecuTorch scaled, we realized that:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Multiple companies want to invest in ExecuTorch as a neutral, shared layer in their on-device AI stack.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Hardware vendors and platform providers need a clear and transparent way to influence direction and contribute.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">The project needs a governance structure that outlives any single organization and keeps ExecuTorch vendor-neutral and open.</span></li>
</ul>
<p><span style="font-weight: 400;">Becoming a PyTorch core project under the PyTorch Foundation gives ExecuTorch:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Vendor-neutral governance with the Foundation’s governing board and charters.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Clear IP, trademark, and branding stewardship, independent of any single company.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Proven open source structures for membership, working groups, and strategic initiatives.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">A natural home alongside adjacent projects in the PyTorch ecosystem.</span></li>
</ul>
<p><span style="font-weight: 400;">Meta will remain a major contributor and a key community member, but no single company will control ExecuTorch’s business governance. The PyTorch Foundation’s experience hosting large, multi-stakeholder projects gives ExecuTorch the right blend of structure and flexibility for this next stage as a PyTorch core project.</span></p>
<h2><span style="font-weight: 400;">Strengthening technical governance</span></h2>
<p><span style="font-weight: 400;">Since its inception, ExecuTorch has operated under a community-driven open source model: maintainers and contributors working across components such as model conversion, runtimes, kernels, backends, and tooling. Responsibilities have been tied to individuals, not just their employers, and we’ve followed the spirit and many of the practices of the PyTorch ecosystem. As ExecuTorch grows, we need more explicit, transparent technical governance to scale responsibly.</span></p>
<p><span style="font-weight: 400;">The ExecuTorch Technical Governance will be as follows:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">The project will adhere to the already existing hierarchical technical governance structure of PyTorch. </span><a href="https://docs.pytorch.org/docs/main/community/persons_of_interest.html#core-maintainers"><span style="font-weight: 400;">Core PyTorch maintainers</span></a><span style="font-weight: 400;"> will oversee larger cross-cutting changes while existing </span><a href="https://docs.pytorch.org/docs/main/community/persons_of_interest.html#executorch-edge-mobile"><span style="font-weight: 400;">Module maintainers</span></a><span style="font-weight: 400;"> will oversee ExecuTorch specific changes. The maintainer membership will be individual and merit-based.</span></li>
</ul>
<p><span style="font-weight: 400;">In the coming weeks, we will:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Publish clear, documented technical decision-making processes, proposals, and escalation paths</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Alignment with familiar open-source patterns (e.g., RFC / proposal processes, release management, standards for compatibility and deprecation)</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Invest in shared CI/CD infrastructure for hardware partners to test and validate their backends</span><span style="font-weight: 400;"><br />
</span></li>
</ul>
<p><span style="font-weight: 400;">This does not fundamentally change how contributors build ExecuTorch day to day. Instead, it adds clarity, predictability, and openness, which are essential for a project that aims to be the neutral, shared runtime layer for on-device AI across the industry.</span></p>
<h2><span style="font-weight: 400;">What’s next</span></h2>
<p><span style="font-weight: 400;">As ExecuTorch become a PyTorch core project, our priorities are:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Growing a diverse contributor and maintainer base across companies, hardware vendors, and independent developers.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Deepening the integration with PyTorch for model export, quantization, and deployment flows.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Expanding hardware and platform coverage so ExecuTorch can run efficiently wherever developers need it—on mobile devices, XR headsets, edge boxes, and embedded systems.</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Continuing to invest in documentation, tooling, and examples to make on-device AI development with ExecuTorch as accessible as possible.</span></li>
</ul>
<p>Thank you.</p>
