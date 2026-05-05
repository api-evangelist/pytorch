---
title: "Optimizing Effective Training Time for Meta’s Internal Recommendation/Ranking Workloads"
url: "https://pytorch.org/blog/optimizing-effective-training-time-for-metas-internal-recommendation-ranking-workloads/"
date: "Fri, 17 Apr 2026 16:00:18 +0000"
author: "Ruilin Chen, Yuzhen Huang, Hang Qi, Mingming Ding, Damian Reeves, Boris Sarana, Kevin Tang, Satendra Gera, Gagan Jain, Sahil Shah, Oguz Ulgen, Mayank Garg, Meet Vadakkanchery, James March, Sophie Lin, Wei Sun"
feed_url: "https://pytorch.org/feed/"
---
<h2><span style="font-weight: 400;">Motivation and Introduction</span></h2>
<p><span style="font-weight: 400;">Across the industry, teams training and serving large AI models face aggressive ROI targets under tight compute capacity. As workloads scale, improving infrastructure effectiveness gets harder because end-to-end runtime increasingly includes overheads beyond “real training” (initialization, orchestration, checkpointing, retries, failures, and recovery). </span></p>
<p><span style="font-weight: 400;">Meta utilizes Effective Training Time (ETT%) to quantify efficiency, defining it as the percentage of total end-to-end (E2E) wall time dedicated to productive training. This metric directly points to areas where time is wasted, thus facilitating the prioritization of efficiency improvements.</span></p>
<p><span style="font-weight: 400;">In this work stream, while grounded in Meta’s production experience using PyTorch for model training, we aim to share broadly useful lessons: some improvements have been implemented in open source—e.g., TorchRec sharding plan improvements and PyTorch 2 (PT2) compilation optimizations that reduce compile time and recompilation—while others (like checkpointing and model publishing) are more Meta-specific, but address common industry bottlenecks and can be adapted elsewhere.</span></p>
<h2><span style="font-weight: 400;">Effective Training Time Definition</span></h2>
<p><span style="font-weight: 400;">Effective Training Time (ETT%) is defined as the percentage of E2E wall time spent on consuming new data. Since the end to end wall time depends on many factors such as model architecture, complexity, training data volume etc, it is hard to directly measure Effective Training Time(ETT%). Instead, focus on measuring idleness and failures, which can be represented as following formula:  </span></p>
<p><img alt="" class="size-full wp-image-63659" height="316" src="https://pytorch.org/wp-content/uploads/2026/04/formula-scaled.jpg" width="2560" /></p>
<p><span style="font-weight: 400;">A visual view of the formula is shown below with three L1 sub-metrics: </span></p>
<ul>
<li style="font-weight: 400;"><b>Time to Start </b><span style="font-weight: 400;">: the period from when a job is allocated hardware to when it begins training the first batch of data.</span></li>
<li style="font-weight: 400;"><b>Time to Recover</b><span style="font-weight: 400;">: the duration required for a training job to restart and resume productive training after a failure or interruption.</span></li>
<li style="font-weight: 400;"><b>Number of Failures</b><span style="font-weight: 400;">: refers to the total count of infra-related interruptions or unsuccessful attempts that occur during the lifecycle of a training job.</span></li>
</ul>
<p><span style="font-weight: 400;">Time to Start and Time to Recover are used to measure the idleness of each single attempt from the system optimization perspective and Number of Failure is targeted to measure different kinds of failures from the reliability area. </span></p>
<p><img alt="" class="alignnone size-full wp-image-63627" height="583" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-9-1.png" width="2048" /></p>
<p style="text-align: center;"><span style="font-weight: 400;">Figure 1. </span><span style="font-weight: 400;">Training Cycle Overview</span></p>
<p><span style="font-weight: 400;">where the definitions for those L2 area are:  </span></p>
<ul>
<li style="font-weight: 400;"><b>Scheduling Time</b><span style="font-weight: 400;">: time spent in infra to get a training job scheduled when resources are available. </span></li>
<li style="font-weight: 400;"><b>Hardware Setup Time</b><span style="font-weight: 400;">: time spent to bring up launcher/trainer binaries in the hardware.</span></li>
<li style="font-weight: 400;"><b>Launcher Init Time</b><span style="font-weight: 400;">: time to start the launcher to enter into the PT2 compilation stage. </span></li>
<li style="font-weight: 400;"><b>PT2 Compilation Time</b><span style="font-weight: 400;">: time to apply PT2 compilation to optimize train model before starting to consume training data. </span></li>
<li style="font-weight: 400;"><b>Effective Training Time</b><span style="font-weight: 400;">: training on time on training data. </span></li>
<li style="font-weight: 400;"><b>Wasted Training Time</b><span style="font-weight: 400;">: time within the train loop but not consuming new training data such as repeated training on samples and blocked training time etc.</span></li>
<li style="font-weight: 400;"><b>Shutdown Time</b><span style="font-weight: 400;">: time to stop a training job. </span></li>
</ul>
<h2><span style="font-weight: 400;">The Journey to Improve ETT% in Meta</span></h2>
<p><span style="font-weight: 400;">Starting from H2’ 24, we have been proactively analyzing the fleetwide Effective Training Time (ETT). This effort aims to establish the ETT% status, identify key focus areas, and implement improvements. </span></p>
<p><span style="font-weight: 400;">For past years, we have developed </span><b>more than 40 new technologies</b><span style="font-weight: 400;"> in order to improve the overall ETT%. The following diagram shows a brief view on improvement in </span><b>Time to Start </b><span style="font-weight: 400;">for each main area:</span></p>
<p><img alt="" class="size-full wp-image-65684" height="1276" src="https://pytorch.org/wp-content/uploads/2026/04/Screenshot-2026-04-15-at-8.51.43-AM.jpg" width="2122" /></p>
<p style="text-align: center;"><span style="font-weight: 400;">Figure 2. </span><span style="font-weight: 400;">Time to Start Improvement Over Each Techs</span></p>
<p><span style="font-weight: 400;">With the team&#8217;s concentrated efforts, we achieved a major milestone by the end of &#8217;25, successfully increasing the Effective Training Time (ETT%) percentage to </span><b>&gt;90% </b><span style="font-weight: 400;">for offline training.</span></p>
<h2><span style="font-weight: 400;">Technique Deep-Dives</span></h2>
<p><span style="font-weight: 400;">The team conducted a detailed analysis of each area contributing to the Effective Training Time (ETT%) and focused optimizations primarily on the following initiatives:</span></p>
<ul>
<li style="font-weight: 400;"><b>Time to Start and Recover:</b><span style="font-weight: 400;"> Optimized trainer initialization and PT2 compilation to lower training costs related to Time to Start and Time to Recover metrics.</span></li>
<li style="font-weight: 400;"><b>Checkpoint Management:</b><span style="font-weight: 400;"> Improved checkpoint processes to minimize idleness during training and reduce unsaved training time.</span></li>
<li style="font-weight: 400;"><b>Shutdown Time Optimizations:</b><span style="font-weight: 400;"> Switched to using CPU machines instead of GPUs for model publishing for inference, resulting in savings on GPU hours for jobs’ shutdown time.</span></li>
<li style="font-weight: 400;"><b>Failure Reduction and Observability:</b><span style="font-weight: 400;"> Collaborated with partner teams to reduce scheduling time and improve the preemption job ratio and established component-level observability and refined the categorization of trainer errors to reduce the frequency of failures.</span></li>
</ul>
<h3><span style="font-weight: 400;">Trainer Initialization Optimizations</span></h3>
<p style="text-align: center;"><span style="font-weight: 400;"><img alt="" class="alignnone size-full wp-image-63635" height="280" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-7-1.png" width="2048" />Figure 3. Trainer Initialization Overview</span></p>
<p><span style="font-weight: 400;">Trainer initialization comprises multiple sub-stages: <code>device_init</code>, <code>process_group_init</code>, <code>preproc_creation</code>, <code>train_module_creation</code>, <code>init_plugins</code>, <code>pre_train</code>, and <code>get_first_batch_data</code>.</span></p>
<p><span style="font-weight: 400;">Beginning in 2024, we have focused on various initiatives to minimize trainer initialization time. The main methodology we applied is</span></p>
<ol>
<li style="font-weight: 400;"><b>Communication optimizations</b><span style="font-weight: 400;">:  remove unnecessary creations or communications between each rank to reduce the overhead cost.</span></li>
<li style="font-weight: 400;"><b>Pipeline Optimizations</b><span style="font-weight: 400;">:  for independent processes, run the sub-stage to overlap with each other to maximize the time usage. </span></li>
</ol>
<h4><span style="font-weight: 400;">Communication Optimizations</span></h4>
<p><span style="font-weight: 400;">Before this work stream, there were numerous unnecessary creations of process groups and non-optimistic communication across different ranks in each job initialization, which collectively contribute to an increase in train initialization time. </span></p>
<p><span style="font-weight: 400;">For instance, instead of relying on numerous </span><code>all_gather</code><span style="font-weight: 400;"> calls to build shard metadata piece by piece—a method that caused substantial overhead in the sharding process—the team implemented an optimization. They now have each rank build its section of the global rank using metadata that is already locally available after the sharding plan broadcast. This change significantly improved sharding time.</span></p>
<p style="text-align: center;"><span style="font-weight: 400;"><img alt="" class="alignnone size-full wp-image-63636" height="934" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-6-1.png" width="2032" />Figure 4. Communication Optimizations Overview</span></p>
<h4><span style="font-weight: 400;">Pipeline Optimizations</span></h4>
<p><span style="font-weight: 400;">Many sub-stages in trainer initialization don&#8217;t have dependencies between each other, which allows the room to create separate processes to run the sub-stage to overlap with each other. </span></p>
<p><span style="font-weight: 400;">For example, the PT2 compilation and DPP warm-up (</span><i>data process we used to fetch training data</i><span style="font-weight: 400;">) to get the first batch of data, are costly and time-consuming steps that occur before the actual training begins. Currently, the PT2 compilation is delayed, as it can only start once the first batch of real data is available for the compilation process.</span></p>
<p><span style="font-weight: 400;">In order to enhance the efficiency of this process, we introduced the new technologies to use the fast batch to quickly get the data which allows PT2 to start compiling much earlier while DPP is still fetching the first batch’ data.</span></p>
<p style="text-align: center;"><span style="font-weight: 400;"><img alt="" class="alignnone size-full wp-image-63639" height="922" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-5-1.png" width="2048" />Figure 5. PT2 compilation and DPP warm-up Parallel</span></p>
<p><span style="font-weight: 400;">This new technology is most beneficial for larger models, such as Foundation Models, because their data loading process is significantly more time-consuming than for other model types. </span></p>
<h3><span style="font-weight: 400;">PT 2.0 Compilation Optimizations</span></h3>
<p><a href="https://pytorch.org/get-started/pytorch-2-x/"><b>PyTorch 2.0</b></a><b> (PT2)</b><span style="font-weight: 400;"> compilation time is another big area where the team invested into. There are 3 main methods we are approaching to reduce the long PT2 compilation time: </span></p>
<ol>
<li style="font-weight: 400;"><span style="font-weight: 400;">Reduce unnecessary recompilations</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Improve overall PT2 cache hit and coverage</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Reduce large amounts of user defined autotune kernels’ configs</span></li>
</ol>
<p><span style="font-weight: 400;">Previously, the team already posted the experience in reducing PT2 compilation time for meta internal workloads, here we just recap the main approaches we did recently and for more details pls refer to the </span><a href="https://pytorch.org/blog/experience-in-reducing-pt2-compilation-time-for-meta-internal-workloads/"><span style="font-weight: 400;">blog</span></a><span style="font-weight: 400;">. </span></p>
<h4><span style="font-weight: 400;">Reduce unnecessary recompilations</span></h4>
<p><span style="font-weight: 400;">Recompilation due to</span><a href="https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html"> <span style="font-weight: 400;">dynamic shapes</span></a><span style="font-weight: 400;"> is a significant source of overhead in our Meta workloads. This recompilation contributes substantially to the overall compilation time across the fleet, resulting in considerable cumulative cost.</span></p>
<p><span style="font-weight: 400;">To address this, the v-team collaborated with the Pytorch team in H1 &#8217;25 to develop </span><a href="https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html"><i><span style="font-weight: 400;">TORCH_COMPILE_DYNAMIC_SOURCES</span></i></a><span style="font-weight: 400;">, which improved the handling of dynamic shapes for parameters by providing an easy and user-friendly way to mark parameters as dynamic without modifying the underlying code. This feature also supports marking integers as dynamic and allows the use of regular expressions to include a broader range of parameters, enhancing flexibility and reducing compilation time.</span></p>
<p><img alt="" class="alignnone size-full wp-image-63640" height="915" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-4-1.png" width="2048" /></p>
<p style="text-align: center;"><span style="font-weight: 400;">Figure 6. Internal Tool to Identify Dynamic Shape</span></p>
<h4><span style="font-weight: 400;">Improve PT2 Cache</span></h4>
<p><a href="https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html#torch-compile-end-to-end-caching-mega-cache"><span style="font-weight: 400;">MegaCache</span></a><span style="font-weight: 400;"> brings together several types of PT2 compilation caches—including components like inductor (the core PT2 compiler), triton bundler (for GPU code), AOT Autograd (for efficient gradient computation), Dynamo PGO (profile-guided optimizations), and autotune settings—into a single archive that can be easily downloaded and shared. </span></p>
<p><span style="font-weight: 400;">By consolidating these elements, MegaCache offers those improvements:</span></p>
<ul>
<li style="font-weight: 400;"><span style="font-weight: 400;">Minimizes repeated requests to remote servers</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Cuts down on time spent setting up models</span></li>
<li style="font-weight: 400;"><span style="font-weight: 400;">Makes startup and retried jobs more dependable, even in distributed or cloud environments</span></li>
</ul>
<p><span style="font-weight: 400;">By the end of 2025, teams worked together to enable the mega cache across all the training platforms. The average PT2 compile time was significantly reduced by approximately </span><b>40% </b><span style="font-weight: 400;">due to this effort.</span></p>
<h4><span style="font-weight: 400;">Autotune config pruning</span></h4>
<p><span style="font-weight: 400;">Autotune in PyTorch 2.0 is a feature that automatically optimizes the performance of PyTorch models by tuning various hyperparameters and settings. With the increasing adoption of </span><a href="https://openai.com/index/triton/"><span style="font-weight: 400;">Triton kernels</span></a><span style="font-weight: 400;">, the time required to compile and search for the best settings and hyperparameters for Triton kernels has increased. </span></p>
<p><span style="font-weight: 400;">To address this, we developed a process to identify the most time-consuming kernels and determine optimal runtime configurations for implementation in the codebase. This approach has led to a substantial reduction in compilation time.</span></p>
<h3><span style="font-weight: 400;">Checkpoint Management</span></h3>
<p><b>Checkpoint</b><span style="font-weight: 400;">: a checkpoint is a saved snapshot of a model’s state during training, including its parameters, optimizer settings, and progress. </span></p>
<p><span style="font-weight: 400;">At Meta, checkpoints are used to ensure that if a training job is interrupted—due to hardware or software issues—the process can resume from the last saved point rather than starting over.</span></p>
<p><span style="font-weight: 400;">Checkpoint saving, while necessary, currently blocks GPU training by demanding memory resources, leading to GPU idle time. Furthermore, the time interval between checkpoint saves directly impacts the amount of training progress that is lost (unsaved training time) if a failure occurs.</span></p>
<p><span style="font-weight: 400;">To address these inefficiencies, the team successfully developed and implemented </span><b>Async Checkpointing</b><span style="font-weight: 400;"> and </span><b>PyTorch Native Staging</b><span style="font-weight: 400;">. These advancements have significantly improved checkpointing performance by reducing the checkpoint blocking time for all models.</span></p>
<p><b>Async checkpointing</b><span style="font-weight: 400;">: it involves creating a copy of the checkpoint in CPU memory, allowing the main trainer process to resume the training loop while a background process completes the checkpoint upload.</span></p>
<p><a href="https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html"><b>PyTorch native staging</b></a><span style="font-weight: 400;">: the initial async checkpoint implementation used custom C++ staging, which was designed to minimize trainer memory usage during staging by utilizing streaming copy. The checkpointing team has developed a separate async checkpointing solution using PyTorch native staging APIs which allows improved save blocking time at the cost of increased trainer memory consumption.</span></p>
<p><span style="font-weight: 400;">These improvements were achieved by significantly reducing the total daily GPU hours blocked for checkpointing.</span></p>
<h4><span style="font-weight: 400;">Reducing Wasted Training Time</span></h4>
<p><span style="font-weight: 400;">Optimizing the time required to save checkpoints directly boosts the Effective Training Time (ETT) percentage by reducing interruptions to the training loop. Furthermore, these checkpoint save improvements can unlock greater ETT% gains when paired with adjustments to the checkpoint interval.</span></p>
<p><span style="font-weight: 400;">Adjusting the checkpoint interval impacts two components of wasted training time:</span></p>
<p><b>Unsaved Training Time</b><b>: </b><span style="font-weight: 400;">this is the training progress lost after a job failure, as any work completed since the last checkpoint is discarded.</span></p>
<ul>
<li style="font-weight: 400;"><em><span style="font-weight: 400;">Calculation</span></em><i><span style="font-weight: 400;">:</span></i><span style="font-weight: 400;"> (# train loop failures) * (checkpoint interval)/2</span></li>
</ul>
<p><b>Checkpoint Save Blocking Time: </b><span style="font-weight: 400;">this is the time the training loop is paused specifically while a new checkpoint is being created.</span></p>
<ul>
<li style="font-weight: 400;"><em><span style="font-weight: 400;">Calculation</span></em><i><span style="font-weight: 400;">:</span></i><span style="font-weight: 400;"> ((time spent in train loop) / (checkpoint interval)) * (blocking time per checkpoint)</span></li>
</ul>
<p><span style="font-weight: 400;">With the job failure rate, the checkpoint interval can be tuned to minimize the expected wasted training time, equal to:</span></p>
<p style="text-align: center;"><em><span style="font-weight: 400;">sum(unsaved training time, checkpoint save blocking time)</span></em></p>
<p><span style="font-weight: 400;">The following graph illustrates the relationship between checkpoint save intervals and the percentage of wasted training time (WTT%), using a hypothetical scenario with a 15-second checkpoint save blocking time and 3 daily failures.</span></p>
<p style="text-align: center;"><span style="font-weight: 400;"><img alt="" class="alignnone size-full wp-image-63641" height="620" src="https://pytorch.org/wp-content/uploads/2026/04/unnamed-3-1.png" width="863" />Figure 7. Checkpoint Save Interval vs Wasted Training Time</span></p>
<p><span style="font-weight: 400;">By optimizing the checkpoint saving interval, the team successfully reduced the unsaved training time for both production and exploration jobs.</span></p>
<h3><span style="font-weight: 400;">Shutdown Time Optimizations</span></h3>
<p><span style="font-weight: 400;">The team dived into each component of the shutdown phase, and found that the model publish processing (model publishing for inference) dominated the post-train process duration.</span></p>
<p><b>Model Publish Processing</b><span style="font-weight: 400;">: </span><span style="font-weight: 400;">Model publishing is the process of optimizing a model using processing code to create an inference-ready snapshot to serve inference. </span></p>
<p><span style="font-weight: 400;">The team&#8217;s analysis led to the adoption of a standalone publishing strategy, which decouples publishing from the training process. With this approach, publishing is initiated only after the training job has finished and created an anchor checkpoint. This checkpoint is then used by a model processing job, leveraging the stored data, to generate the final inference-ready snapshot.</span></p>
<p><span style="font-weight: 400;">The key differences between this standalone publishing method and the traditional &#8220;trending end&#8221; model publishing are visually represented in the diagram below.</span></p>
<p><img alt="" class="size-full wp-image-65680" height="930" src="https://pytorch.org/wp-content/uploads/2026/04/Screenshot-2026-04-15-at-8.50.13-AM-scaled.jpg" width="2560" /></p>
<p style="text-align: center;"><span style="font-weight: 400;">Figure 8.  &#8220;Trending End&#8221; Model Publish vs Standalone Publish</span></p>
<p><span style="font-weight: 400;">The implementation of the new model publishing pipeline has successfully shortened the shutdown time for each job by approximately </span><b>30 minutes</b><span style="font-weight: 400;">.</span></p>
<p><img alt="" class="size-full wp-image-63646" height="964" src="https://pytorch.org/wp-content/uploads/2026/04/Screenshot-2026-04-11-at-8.07.56-AM.jpg" width="2392" /></p>
<h3><span style="font-weight: 400;">Failure Reduction and Observability</span></h3>
<p><span style="font-weight: 400;">A major focus area for the team has been failure reduction, as the number of failures significantly impacts the overall Effective Training Time (ETT) percentage. Regressions from code or configuration changes can directly cause this percentage to drop.</span></p>
<p><span style="font-weight: 400;">Fluctuations in the ETT dashboard are primarily attributed to two factors:</span></p>
<ol>
<li style="font-weight: 400;"><b>Increased Job Preemptions:</b><span style="font-weight: 400;"> A higher volume of running jobs leads to more preemptions.</span></li>
<li style="font-weight: 400;"><b>Service Regressions:</b><span style="font-weight: 400;"> Issues with services cause a greater number of job failures.</span></li>
</ol>
<p><span style="font-weight: 400;">To tackle preemptions, we are collaborating with infrastructure teams to develop a new scheduling algorithm aimed at lowering the preemption ratio without negatively affecting users&#8217; quotas or experience.</span></p>
<p><span style="font-weight: 400;">Regarding failure reduction, a dedicated team is scrutinizing each ETT-related component and building dashboards to monitor overall ETT performance, including Time to Start/Time to Restart (TTS/TTR), unsaved training time, and checkpoint saving time. This proactive monitoring ensures that any regression is detected and mitigated early within the SLA.</span></p>
<h2><span style="font-weight: 400;">In the End</span></h2>
<p><span style="font-weight: 400;">As model training scales, resource constraints are becoming a defining challenge across the industry. For years, a major lever for improving training efficiency has been increasing Model FLOPs Utilization (MFU) through techniques like model co-design and kernel optimization. That work remains essential, but large-scale training has surfaced a complementary bottleneck: significant GPU time is spent idle outside the steady-state training loop.</span></p>
<p><span style="font-weight: 400;">Our analysis shows that non-training overhead can be substantial especially on some of the largest runs. </span></p>
<p><span style="font-weight: 400;">To address this, we launched a successful workstream focused on improving</span><b> Effective Training Time (ETT%)</b><span style="font-weight: 400;">, which has already produced meaningful capacity savings. The key takeaway for practitioners is simple: to improve cost and throughput at scale, you must optimize the “in-between” phases—not just the training steps.</span></p>
<p><span style="font-weight: 400;">Since our training stack utilizes PyTorch, we made an effort to ensure these enhancements are applicable beyond a single environment. We have open-sourced and shared relevant building blocks, such as those in TorchRec and PyTorch 2, within the open-source PyTorch ecosystem. This allows others to leverage these improvements, replicate our results, and build upon our work. Other components, like model publishing and checkpointing, are more specific to Meta but tackle common industry challenges and can be adapted for use elsewhere.</span></p>
<p><span style="font-weight: 400;">We hope these lessons help teams diagnose similar bottlenecks, apply ETT%-style measurement, and contribute further improvements back to the ecosystem.</span></p>
<h2><span style="font-weight: 400;">Acknowledgements</span></h2>
<p><span style="font-weight: 400;">We extend our gratitude to </span><span style="font-weight: 400;">Max Leung</span><span style="font-weight: 400;">, </span><span style="font-weight: 400;">Apoorv Purwar</span><span style="font-weight: 400;">, </span><span style="font-weight: 400;">Musharaf Sultan</span><span style="font-weight: 400;">, </span><span style="font-weight: 400;">John Bocharov</span><span style="font-weight: 400;">, </span><span style="font-weight: 400;">Barak Pat</span><span style="font-weight: 400;">, </span><span style="font-weight: 400;">Jonathan Tang</span><span style="font-weight: 400;">, </span><span style="font-weight: 400;">Vivek Trehan</span><span style="font-weight: 400;">, </span><span style="font-weight: 400;">Chris Gottbrath</span><span style="font-weight: 400;"> and </span><span style="font-weight: 400;">Vitor Brumatti Pereira</span><span style="font-weight: 400;">  for their valuable reviews and insightful support. We also thank the entire Meta team responsible for the development and productionization of this workstream.</span></p>
