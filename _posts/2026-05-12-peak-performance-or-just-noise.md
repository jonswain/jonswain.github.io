---
layout: post
title: "Peak Performance or Just Noise?"
date: 2026-05-13 14:00:00 
permalink: /peak-performance-or-just-noise/
redirect_from:
    - /ai/cheminformatics/data%20science/machine%20learning/openadmet/omsf/2026/05/13/peak-performance-or-just-noise.html
description: "Why leaderboard rank gaps can be noise, and how paired bootstrap intervals and permutation tests improve blind challenge comparisons."
categories:
    - ai
    - cheminformatics
    - data-science
    - machine-learning
    - open-science
---

Sorry it's been quiet on here for the past 6 months. I've been lucky enough to start working at [OpenADMET](https://openadmet.org), part of the [Open Molecular Software Foundation (OMSF)](https://omsf.io). Part of my work has been on blind challenges and I've written a blog post for OpenADMET on leaderboard analysis.

[You can find the blog post here.](https://openadmet.ghost.io/peak-performance-or-just-noise)

## Statistical Comparison of Blind Challenge Entries

When we look at machine learning leaderboards, it's easy to treat raw scores as definitive rankings. However, in small-molecule drug discovery and blind challenges like the OpenADMET-ExpansionRx challenge, the performance margins between top entries are often smaller than the underlying noise of the data.

In this post, I dig into the statistical pitfalls of standard ranking approaches (and how easily we can accidentally "*p*-hack" our benchmarks using standard bootstrap methods). I then evaluate robust alternatives, specifically **Paired Bootstrap Confidence Intervals** and **Permutation Testing**, to show how we can reliably separate genuine model superiority from pure luck.

It ended up being quite long, so I'd recommend making a hot drink before you settle down to read the whole thing!
