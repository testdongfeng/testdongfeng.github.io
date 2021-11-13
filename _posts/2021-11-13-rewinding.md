---
layout: post
title: Rewinding as an Alternative to Finetuning for Network Pruning
tags: []
authors: Shuqin Xie, Dongfeng Yu
---
## Introducation
This blog post is about our thoughts on the topics mentioned by the ICLR paper [Comparing Rewinding and Fine-tuning in Neural Network Pruning](#Renda), which we ill be refering to as "this paper". We hope you enjoy reading our article.
## Network Pruning
The efficiency of neural network inference is of importance for both the research community and industry since it's critical for the real-world applications of deep learning. Network pruning is an active research field involving various ways to reduce the size of a given model, for the benefit of memory consumption and computation cost. One of the early works ([Han, Song, et al.](#Han)) considers a simple yet effective way to prune an already trained neural network: Set all the weights whose magnitude is under a certain threshold to zero and fine-tune the resulting network. Intuitively, the weights that are close to zero have less impact on the overall performance of the network, thus removing them would only cause a small drop in accuracy, which will them be remedied with fine tuning. It is also showed that doing this pruning step for multiple times instead of pruning directly to the desired sparsity can increase the resulting accuracy as well. So in summary, a typical pruning pipeline would look like:
![Pipeline]({{site.url}}/public/images/pipeline.jpg)
Since then, many following works, for example, [Molchanov, P., et al.](#Molchanov), explores different ways to improve the pruning step in this pipeline. Instead of global pruning based solely on the magnitude of weights, it's possible to achieve better results by considering better criterions to choose the connections to remove. For example, we can approximate the change to the loss function incurred by setting a weight to zero by Taylor series. However, this paper shows us that the finetuning step, which seems to be a standard procedure, can also be replaced with more advanced techniques.
## Evaluation Metrics
As we have now defined the problem, we can now talk about the evaluation metrics.
### Accuracy
Accuracy is of course the metric that we care about. Ideally, we expect the pruned the model to have simliar performance to the original one. Here accuracy is a general term refer to any metric that is used to evaluate the prediction of the model, for example, classification accuracy, mAP, F1 score, or mean squared error.
## The Lottery Ticket Hypothesis
As we have mentioned, this paper provides alteratives to finetuning, but before we dive into them, we would like to first talk about the lottery ticket hypothesis ([Frankle, Jonathan, and Michael Carbin.](#Frankle)), which is the motivation behind these alternatives. The lottery ticket hypothesis claims that a random initialized neural network contains a sparse subnet which can be trained from scratch to reach test accruacy simliar to the full network. The way to identify this subnet is quite simliar to the pruning pipeline. A model is trained and pruned. However, then the surviving weights are rewound to their initialization to get the winning ticket. Then this subnet can be trained to verify the hypothesis.
## References
<a name="Renda">Renda, Alex, Jonathan Frankle, and Michael Carbin. "Comparing Rewinding and Fine-tuning in Neural Network Pruning." International Conference on Learning Representations. 2019.</a>

<a name="Han">Han, Song, et al. "Learning both Weights and Connections for Efficient Neural Network." Advances in Neural Information Processing Systems 28 (2015).</a>

<a name="Molchanov">Molchanov, P., et al. "Pruning convolutional neural networks for resource efficient inference." 5th International Conference on Learning Representations, ICLR 2017-Conference Track Proceedings. 2019.</a>

<a name="Frankle">Frankle, Jonathan, and Michael Carbin. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." International Conference on Learning Representations. 2018.</a>
