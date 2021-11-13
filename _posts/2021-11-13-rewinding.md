---
layout: post
title: The Lottery Ticket Hypothesis and It’s Application on Model Pruning
tags: []
authors: Shuqin Xie, Dongfeng Yu
---
## Introducation
This blog post is about our thoughts on the topics mentioned by the ICLR paper [Comparing Rewinding and Fine-tuning in Neural Network Pruning](#Renda). We hope you enjoy reading it.
## A Typical Pruning Pipeline
The efficiency of neural network inference is of importance for both the research community and industry since it's critical for the real-world applications of deep learning. One of the early works ([Han, Song, et al.](#Han)) considers a simple yet effective way to prune an already trained neural network: Set all the weights whose magnitude is under a certain threshold to zero and fine-tune the resulting network. Intuitively, the weights that are close to zero have less impact on the overall performance of the network, thus removing them would only cause a small drop in accuracy, which will them be remedied with fine tuning. It is also showed that doing this pruning step for multiple times instead of pruning directly to the desired sparsity can increase the resulting accuracy as well. So in summary, a typical pruning pipeline would look like:
![Pipeline]({{site.url}}/public/images/pipeline.jpg)
## References
<a name="Renda">Renda, Alex, Jonathan Frankle, and Michael Carbin. "Comparing Rewinding and Fine-tuning in Neural Network Pruning." International Conference on Learning Representations. 2019.</a>
<a name="Han">Han, Song, et al. "Learning both Weights and Connections for Efficient Neural Network." Advances in Neural Information Processing Systems 28 (2015).</a>