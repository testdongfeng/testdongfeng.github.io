---
layout: post
title: Rewinding as an Alternative to Fine-tuning for Network Pruning
tags: []
authors: Shuqin Xie, Dongfeng Yu
---
## Introducation
This blog post is about our thoughts on the topics mentioned by the ICLR paper [Comparing Rewinding and Fine-tuning in Neural Network Pruning](#Renda), which we will be refering to as "this paper". We hope you enjoy reading our article.
## Network Pruning
The efficiency of neural network inference is of importance for both the research community and industry since it's critical for the real-world applications of deep learning. Network pruning is an active research field involving various ways to reduce the size of a given model, for the benefit of memory consumption and computation cost. One of the early works ([Han, Song, et al.](#Han)) considers a simple yet effective way to prune an already trained neural network: Set all the weights whose magnitude is under a certain threshold to zero and fine-tune the resulting network. Intuitively, the weights that are close to zero have less impact on the overall performance of the network, thus removing them would only cause a small drop in accuracy, which will them be remedied with fine tuning. It is also showed that doing this pruning step for multiple times instead of pruning directly to the desired sparsity can increase the resulting accuracy as well. So in summary, a typical pruning pipeline would look like:
![Pipeline]({{site.url}}/public/images/pipeline.jpg)
Since then, many following works, for example, [Molchanov, P., et al.](#Molchanov), explores different ways to improve the pruning step in this pipeline. Instead of global pruning based solely on the magnitude of weights, it's possible to achieve better results by considering better criterion to choose the connections to remove. For example, we can approximate the change to the loss function incurred by setting a weight to zero by Taylor series. However, this paper shows us that the fine-tuning step, which seems to be a standard procedure, can also be replaced with more advanced techniques.
## Evaluation Metrics
As we have already defined the problem, we can now talk about the evaluation metrics.
### Accuracy
Accuracy is of course the metric that we care about. Ideally, we expect the pruned the model to have similiar performance to the original one. Here accuracy is a general term refer to any metric that is used to evaluate the prediction of the model, for example, classification accuracy, F1 score, mAP, or mean squared error.
### Sparsity
We of course also care about sparsity and the trade-off between it and the accuracy. However, sparsity is a vaguely defined term. We can surely define it as the number of zeros in the weights. However, it is not just how many zeros that matters but also the structure of the zeros. A simple example would be, if we can prune several channels away from a CNN, then the acceleration and memory reduce is immediately gained without any hardware or software support. However, if we zero out random entries across all channels and reach the same sparsity, we may not get any advantage over the original model. Specially designed hardware and software ([Han, Song, et al.](#Song)) may be required for these sparse convolution kernels. Below is a visualization showing the two kinds of sparsity:
![Sparsity]({{site.url}}/public/images/sparsity.jpg)
Many previous works, such as [He, Yihui, Xiangyu Zhang, and Jian Sun.](#He) explore the methods for structured pruning. The rewinding methods that we will talk about below can be applied to both kinds of pruning.
### Efficiency
We can also directly evaluate the pruned models based on their inference speed and memory consumption. But of course this is highly dependent on the hardware used, which also implies there may not be a silver bullet. Instead, use the right pruning algorithm for your specific platform of computation.
### Search Cost
We also care about the resources needed to create the pruned model. This is not as important as the previous metrics in a real-world setting, but again we may sometimes need to maintain a reasonable cost.
## The Lottery Ticket Hypothesis
As we have mentioned, this paper provides alternatives to fine-tuning, but before we dive into them, we would like to first talk about the lottery ticket hypothesis ([Frankle, Jonathan, and Michael Carbin.](#Frankle)), which is the motivation behind these alternatives. The lottery ticket hypothesis claims that a random initialized neural network contains a sparse sub-net which can be trained from scratch to reach test accuracy similiar to the full network in less iteration. And the standard pruning pipeline can be used to identify this sub-net. A model is trained and pruned. However, then the surviving weights are rewound to their initialization to get the winning ticket. Then this sub-net can be trained to verify the hypothesis.
## Fine-tuning
This paper compares the proposed methods with fine-tuning, which is a reasonable basedline. Fine-tuning is a commonly used technique in transfer learning to adapt pretrained model to the new data domain. In pruning community, it is first used by ([Han, Song, et al.](#Han)) and later become a paradigm in this domain. After pruning, we usually retrain the pruned network for $t$ epochs with the last epoch's learning rate to recover from the loss of accuracy.
## Weight Rewinding
In the lottery ticket hypothesis paper, the authors discussed, without verification, the possibility of a new retraining technique other than fine-tuning. It is termed weight rewinding and is intended to replace fine-tuning after pruning. The idea is quite simple: instead of using the final trained weight, it rewinds the original weights and learning rate scheduler to $t$ epochs before, then retrains the pruned network for $t$ epochs. Note that the original paper only proposed this possibility without validating it. This paper verified this idea and proved that it is in general in par or better than fine-tuning under the same search cost.
## Learning Rate Rewinding
This paper further proposed a learning rate rewinding technique, that is, retraining using the final weight but rewinding learning rate schedule to $t$ epochs before. It is in the middle of fine-tuning and weight rewinding, where fine-tuning uses the final weight and final learning rate, while weight rewinding reuses previous weights and previous learning rate scheduler. Their experiments show that learning rate rewinding is in general better than weight rewinding also with the same search cost. The whole proposed algorithm looks like this:
```
Algorithm 1: Magnitude-based Pruning with Learning Rate Rewinding
    1. Train the model with standard hyper-parameters.
    2. Fix the weights whose magnitude is among 20% lowest to 0.
    3. Retrain the model but with learning rate rewound to the original training schedule.
    4. Keep doing step 2 and 3 until the expected sparsity is reached.
```
## Reflections
This paper breaks the "common knowledge" that we only uses a fixed learning rate during fine-tuning. It is quite common that people simply follows existing "common knowledge" without thinking whether it is necessary or whether there is better choice. This paper shows that we don't need to blindly use fine-tuning, there exists simple and better solutions called learning rate rewinding. Another interesting thing is that this paper extends the field of network pruning by connecting it with the lottery ticket hypothesis. It's quite amazing the the initialization of the weights at the very beginning of the training may have already decided the good pruned sub-net that we want to find. And with that knowledeg, we can actually find a improved way to do pruning.
## References
<a name="Renda">Renda, Alex, Jonathan Frankle, and Michael Carbin. "Comparing Rewinding and Fine-tuning in Neural Network Pruning." International Conference on Learning Representations. 2019.</a>

<a name="Han">Han, Song, et al. "Learning both Weights and Connections for Efficient Neural Network." Advances in Neural Information Processing Systems 28 (2015).</a>

<a name="Molchanov">Molchanov, P., et al. "Pruning convolutional neural networks for resource efficient inference." 5th International Conference on Learning Representations, ICLR 2017-Conference Track Proceedings. 2019.</a>

<a name="Frankle">Frankle, Jonathan, and Michael Carbin. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." International Conference on Learning Representations. 2018.</a>

<a name="Song">Han, Song, et al. "EIE: Efficient Inference Engine on Compressed Deep Neural Network." 2016 ACM/IEEE 43rd Annual International Symposium on Computer Architecture (ISCA). IEEE Computer Society, 2016.</a>

<a name="He">He, Yihui, Xiangyu Zhang, and Jian Sun. "Channel pruning for accelerating very deep neural networks." Proceedings of the IEEE international conference on computer vision. 2017.</a>