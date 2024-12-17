---
# # Front matter. This is where you specify a lot of page variables.
# layout: default
# title:  "PHLAME"
# date:   2024-02-13 10:00:00 -0500
# description: >- # Supports markdown
#   Pseudospectral Heat fLow using the Affine geoMetric Equation
# show-description: true

# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "PHLAME"
date:   2024-10-17 10:00:00 -0500
description: >- # Supports markdown
  Bring the Heat:  Rapid Trajectory Optimization with Pseudospectral Techniques and the Affine Geometric Heat Flow Equation
show-description: true

# Add page-specific mathjax functionality. Manage global setting in _config.yml
mathjax: false
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: false

# Preview image for social media cards
image:
  path: /assets/phlame_cover_photo.png
  height: 600
  width: 800
  alt: PHLAME Main Figure

# Only the first author is supported by twitter metadata
authors:
  - name: Challen Enninful Adu
    email: enninful@umich.edu
  - name: Cesar E. Ramos Chuquiure
    email: cesarch@umich.edu
  - name: Bohao Zhang
    email: jimzhang@umich.edu
  - name: Ram Vasudevan
    email: ramv@umich.edu

# If you just want a general footnote, you can do that too.
author-footnotes:
  All authors affiliated with the department of Robotics at the University of Michigan, Ann Arbor.

links:
  - icon: arxiv
    icon-library: simpleicons
    text: ArXiv
    url: https://arxiv.org/abs/2411.12962
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/roahmlab/PHLAME
  - icon: bi-file-earmark-text
    icon-library: bootstrap-icons
    text: Supplementary Appendices
    url: PHLAME_Appendices.pdf

# End Front Matter
---

<!-- BEGIN DOCUMENT HERE -->

{% include sections/authors %}
{% include sections/links %}

---

<!-- BEGIN OVERVIEW FIGURE -->
<div class="fullwidth video-container" style="display: flex; flex-wrap:nowrap; gap: 10px; padding: 0 0.2em; justify-content: center">
  <div class="video-item" style="flex: 1 1 50%; max-width: 50%;">
    <video
      class="autoplay-on-load"
      preload="none"
      controls
      disablepictureinpicture
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto;">
      <source src="assets/aghf_digit_stretch.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p>Digit Doing Yoga Pose (Generated in ~3s) </p>
  </div>
  <div class="video-item" style="flex: 1 1 50%; max-width: 50%;">
    <video
      class="autoplay-on-load"
      preload="none"
      controls
      disablepictureinpicture
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto;">
      <source src="assets/aghf_digit_stair.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p>Digit Stepping on Box (Generated in ~2s) </p>
  </div>
</div> <!-- END OVERVIEW VIDEOS -->

<!-- BEGIN ABSTRACT -->
<div markdown="1" class="content-block justify grey">

# [Abstract](#abstract)
Generating optimal trajectories for high-dimensional robotic systems in a time-efficient manner while adhering to constraints is a challenging task. To address this challenge, this paper introduces PHLAME, which applies pseudospectral collocation and spatial vector algebra to efficiently solve the Affine Geometric Heat Flow (AGHF) Partial Differential Equation (PDE) for trajectory optimization. Unlike traditional PDE approaches like the Hamilton-Jacobi-Bellman (HJB) PDE, which solve for a function over the entire state space, computing a solution to the AGHF PDE scales more efficiently because its solution is defined over a two-dimensional domain, thereby avoiding the intractability of state-space scaling. To solve the AGHF one usually applies the Method of Lines (MOL), which works by discretizing one variable of the AGHF PDE, effectively converting the PDE into a system of ordinary differential equations (ODEs) that can be solved using standard time-integration methods. Though powerful, this method requires a fine discretization to generate accurate solutions and still requires evaluating the AGHF PDE which can be commputationally expensive for high dimensional systems. PHLAME overcomes this deficiency by using a pseudospectral method, which reduces the number of function evaluations required to yield a high accuracy solution thereby allowing it to scale efficiently to high-dimensional robotic systems. To further increase computational speed, this paper presents analytical expressions for the AGHF and its Jacobian, both of which can be computed efficiently using rigid body dynamics algorithms. The proposed method PHLAME is tested across various dynamical systems, with and without obstacles and compared to a number of state-of-the-art techniques. PHLAME is able to generate trajectories for a 44-dimensional state-space system in ~5 seconds, much faster than current state-of-the-art techniques. Code is available on GitHub at [roahmlab/PHLAME](https://github.com/roahmlab/PHLAME).

</div> <!-- END ABSTRACT -->

<!-- BEGIN METHOD -->
<div markdown="1" class="justify">

# [Approach](#method)

![link_construction](./assets/phlame_cover_photo.png)
{: class="fullwidth"}

<!-- # Contributions -->
To address the limitations of existing approaches, this paper proposes Pseudospectral Heat fLow using the Affine geoMetric heat flow Equation (PHLAME). 
The proposed method applies pseudospectral collocation and spatial vector algebra to efficiently solve the Affine Geometric Heat Flow (AGHF) Partial Differential Equation (PDE) enabling rapid trajectory optimization for high dimensional systems.
PHLAME works by first taking in some initial guess of a trajectory (shown in dark blue) which does not have to be dynamically feasible and evolves it into some dynamically feasible final trajectory (shown in dark green).
Both trajectories start and end at $$\mathtt{x}(0) = \mathtt{x}_0$$ and $$\mathtt{x}(T) = \mathtt{x}_f$$ respectively.
Notice that at the initial trajectory Digit has a dynamically infeasible set of configurations during it's stepping trajectory and that at the end of the PHLAME solve that trajectory is made into a dynamically feasible one where Digit is able to step over the box.

This paper’s contributions are four-fold:
1. A pseudospectral method that reduces the number of AGHF evaluations and nodes when compared to the classical MOL, which allows PHLAME to scale up to high dimensional robotic systems;
2. An analytical expression for the AGHF in terms of the rigid body dynamics equation and an algorithm to rapidly evaluate this analytical AGHF expression using spatial vector algebra based rigid body dynamics algorithms;
3. An analytical expression for the jacobian of the AGHF and an algorithm to rapidly compute it using spatial vector algebra based rigid body dynamics algorithms;
4. A demonstration that PHLAME generating trajectories for a number of different high-dimensional robots systems with and without constraints in a few seconds, much faster that other state-of-the-art trajectory optimization methods

</div><!-- END METHOD -->

<!-- START RESULTS -->
<div markdown="1" class="content-block grey justify">

# [Results](#simulation-results)
## PHLAME Solve Time Comparison
This section shows the solve of PHLAME against comparison methods Crocoddyl and Original AGHF.
The results show the time each method takes to solve a fixed time swing up problem for a 1-, 2-, 3-, 4-, and 5-link pendulum model (N = 1 to 5), a fixed time and final state specified trajectory optimization for the Kinova arm (N = 7), and a fixed time and final state trajectory optimization having the pinned Digit execute a step (N = 22).

![link_construction](./assets/scalability_t_solve.png)
{: style="width:90%; margin:0 auto; display:block;" }

The bar plot above compares the mean solve times for the three different trajectory generation algorithms: PHLAME, AGHF and Crocoddyl.
Each experiment was run ten times using the best solver parameter set. 
For $$N>=5$$ we do not show results for the Original AGHF implementation as no results were obtained in a reasonable amount of time. 
Overall, we see that PHLAME shows better empirical scalability and solve time than the other methods.

## Kinova Obstacle Avoidance Scenarios
The following videos demonstrate the various trajectories generated by PHLAME for the 7DOF Kinova Gen3 to navigate from the start configuration (turqoise) to the goal configuration (gold) while avoiding obstacles (red).
All of these trajectories were generated in &lt; 0.3 seconds


<!-- START KINOVA CONS VIDEOS -->
<div class="video-container" style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;">
  <div class="video-item" style="flex: 1 1 32%; max-width: 32%;">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      style="display:block; width:100%; height:auto;">
      <source src="assets/kinova_scenario_3_1.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <!-- <p>10 obstacles</p> -->
  </div>
  <div class="video-item" style="flex: 1 1 32%; max-width: 32%;">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      style="display:block; width:100%; height:auto;">
      <source src="assets/kinova_scenario_8_2.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <!-- <p>20 obstacles</p> -->
  </div>
  <div class="video-item" style="flex: 1 1 32%; max-width: 32%;">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      style="display:block; width:100%; height:auto;">
      <source src="assets/kinova_scenario_9_1.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <!-- <p>40 obstacles</p> -->
  </div>
</div><!-- END RANDOM VIDEOS -->


## Digit Scenarios
The following videos demonstrate various trajectories generated by PHLAME for the 22DOF Pinned Digit V3 to navigate from the start configuration (turqoise) to the goal configuration (gold).
All of these trajectories were generated in &lt; 5 seconds

<!-- START DIGIT UNC VIDEOS -->
<div class="fullwidth video-container" style="display: flex; flex-wrap:nowrap; gap: 10px; padding: 0 0.2em; justify-content: center">
  <div class="video-item" style="flex: 1 1 48%; max-width: 48%;">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      style="display:block; width:100%; height:auto;">
      <source src="assets/aghf_digit_stretch.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item" style="flex: 1 1 48%; max-width: 48%;">
    <video
      class="autoplay-in-frame"
      preload="none"
      disableremoteplayback
      disablepictureinpicture
      playsinline
      muted
      loop
      onclick="this.paused ? this.play() : this.pause();"
      style="display:block; width:100%; height:auto;">
      <source src="assets/aghf_digit_step.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
</div><!-- END DIGIT VIDEOS -->
</div><!-- END RESULTS -->

<div markdown="1" class="justify">
  

<div markdown="1" class="content-block grey justify">
  
# [Citation](#citation)

This project was developed in [Robotics and Optimization for Analysis of Human Motion (ROAHM) Lab](http://www.roahmlab.com/) at the University of Michigan - Ann Arbor.

```bibtex
@article{enninfulphlame2024,
  title={Bring the Heat: Rapid Trajectory Optimization With Pseudospectral Techniques and the Affine Geometric Heat Flow Equation},
  author={Challen Enninful Adu and César E. Ramos Chuquiure and Bohao Zhang and Ram Vasudevan},
  year={2024},
  eprint={2411.12962},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2411.12962}}
```
</div>


<!-- below are some special scripts -->
<script>
window.addEventListener("load", function() {
  // Get all video elements and auto pause/play them depending on how in frame or not they are
  let videos = document.querySelectorAll('.autoplay-in-frame');

  // Create an IntersectionObserver instance for each video
  videos.forEach(video => {
    const observer = new IntersectionObserver(entries => {
      const isVisible = entries[0].isIntersecting;
      if (isVisible && video.paused) {
        video.play();
      } else if (!isVisible && !video.paused) {
        video.pause();
      }
    }, { threshold: 0.25 });

    observer.observe(video);
  });

  // document.addEventListener("DOMContentLoaded", function() {
  videos = document.querySelectorAll('.autoplay-on-load');

  videos.forEach(video => {
    video.play();
  });
});
</script>