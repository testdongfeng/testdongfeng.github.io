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
Since then, many following works, for example, [Molchanov, P., et al.](#Molchanov), explores different ways to improve the pruning step in this pipeline. Instead of global pruning based solely on the magnitude of weights, it's possible to achieve better results by considering better criterions to choose the connections to remove. For example, we can approximate the change to the loss function incurred by setting a weight to zero by Taylor series. However, this paper shows us that the finetuning step, which seems to be a standard procedure, can also be replaced with more advanced techniques.
## The Lottery Ticket Hypothesis
Before we dive into the alternatives of finetuning, we would like to first talk about the lottery ticket hypothesis ([Frankle, Jonathan, and Michael Carbin.](#Frankle)). The lottery ticket hypothesis claims that a random initialized neural network contains a sparse subnet which can be trained from scratch to reach test accruacy simliar to the full network. The way to identify this sub-net is quite simliar to the pruning pipeline. A model is trained and pruned. However, then the surviving weights are rewound to their initilazation to get the winning ticket. Then this subnet can be trained to verify the hypothesis.
## References
<a name="Renda">Renda, Alex, Jonathan Frankle, and Michael Carbin. "Comparing Rewinding and Fine-tuning in Neural Network Pruning." International Conference on Learning Representations. 2019.</a>

<a name="Han">Han, Song, et al. "Learning both Weights and Connections for Efficient Neural Network." Advances in Neural Information Processing Systems 28 (2015).</a>

<a name="Molchanov">Molchanov, P., et al. "Pruning convolutional neural networks for resource efficient inference." 5th International Conference on Learning Representations, ICLR 2017-Conference Track Proceedings. 2019.</a>

<a name="Frankle">Frankle, Jonathan, and Michael Carbin. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." International Conference on Learning Representations. 2018.</a>
