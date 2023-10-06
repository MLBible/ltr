---
layout: layout
title: RankNet
---
# Introduction

RankNet is one of the learning-to-rank algorithms used by commercial search engines. In RankNet, the loss function is also defined on a pair of documents, but the hypothesis is defined with the use of a scoring function $$f$$. At a given point during training, RankNet maps an input feature vector $$\mathbf{x} \in \mathbb{R}^n$$ to a number $$f(\mathbf{x})$$. For a given query, each pair of URLs $$U_i$$ and $$U_j$$ with differing labels is chosen, and each such pair (with feature vectors $$\mathbf{x}_i$$ and $$\mathbf{x}_j$$) is presented to the model, which computes the scores $$s_i = f(\mathbf{x}_i)$$ and $$s_j = f(\mathbf{x}_j)$$. Let $$U_i \rhd U_j$$ denote the event that $$U_i$$ should be ranked higher than $$U_j$$ (for example, because $$U_i$$ has been labeled 'excellent' while $$U_j$$ has been labeled 'bad' for this query; note that the labels for the same URLs may be different for different queries). The two outputs of the model are mapped to a learned probability that $$U_i$$ should be ranked higher than $$U_j$$ via a sigmoid function, thus:

$$
P_{ij} \equiv P(U_i \rhd U_j) \equiv \frac{1}{1 + e^{-\sigma(s_i - s_j)}} \tag{1}
$$

where the choice of the parameter $$\sigma$$ determines the shape of the sigmoid. For RankNet, the underlying model can be any model for which the output of the model is a differentiable function of the model parameters. 

# Cost Function for RankNet

Let $$\overline{P}_{ij}$$ be the known probability that training URL $$U_i$$ should be ranked higher than training URL $$U_j$$. For a given query, let $$S_{ij} \in \{0, \pm 1\}$$ be defined as $$1$$ if document $$i$$ has been labeled to be more relevant than document $$j$$, $$-1$$ if document $$i$$ has been labeled to be less relevant than document $$j$$, and 0 if they have the same label. We assume that the desired ranking is deterministically known, so that $$\overline{P}_{ij} = \frac{1}{2}(1 + S_{ij})$$. Then the cross-entropy between the target probability and the modeled probability is used as the loss function:

$$
C = -\overline{P}_{ij} \log(P_{ij}) - (1 - \overline{P}_{ij}) \log(1 - P_{ij})
$$

Alternatively,

$$
C = \frac{1}{2}(1 - S_{ij})\sigma(s_i - s_j) + \log(1 + e^{-\sigma(s_i - s_j)}) \tag{2}
$$


A neural network is then used as the model, and gradient descent is the optimization algorithm to learn the scoring function $$f$$. The cost is symmetric (swapping $$i$$ and $$j$$ and changing the sign of $$S_{ij}$$ should leave the cost invariant): for $$ S_{ij} = 1$$,

$$
C = \log(1 + e^{-\sigma(s_i - s_j)})
$$

while for $$S_{ij} = -1$$, 

$$
C = \log(1 + e^{-\sigma(s_j - s_i)})
$$

Note that when $$s_i = s_j$$, the cost is $$\log(2)$$, so the model incorporates a margin (that is, documents with different labels but to which the model assigns the same scores are still pushed away from each other in the ranking). Also, asymptotically, the cost becomes linear (if the scores give the wrong ranking) or zero (if they give the correct ranking). 

# Learning Algorithm

From (2), we get:

$$
\frac{\partial C}{\partial s_i} = \sigma \left(\frac{1}{2}(1 - S_{ij}) - \frac{1}{1 + e^{\sigma(s_i - s_j)}}\right) = -\frac{\partial C}{\partial s_j}
$$

This gradient is used to update the weights $$w_k \in \mathbb{R}$$ (i.e., the model parameters) to reduce the cost via stochastic gradient descent:

$$
w_k \to w_k - \eta \frac{\partial C}{\partial w_k} = w_k - \eta \left(\frac{\partial C}{\partial s_i} \frac{\partial s_i}{\partial w_k} + \frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k}\right)
$$

where $$\eta$$ is a positive learning rate (a parameter chosen using a validation set, typically in our experiments, $$10^{-3}$$ to $$10^{-5}$$). Explicitly:

$$
\delta C = \sum_k \frac{\partial C}{\partial w_k} \delta w_k = \sum_k \frac{\partial C}{\partial w_k} \left(- \eta \frac{\partial C}{\partial w_k}\right) = -\eta \sum_k \left(\frac{\partial C}{\partial w_k}\right)^2 < 0
$$

A nested ranker is sometimes built on top of RankNet to further improve the retrieval performance. Specifically, the new method iteratively re-ranks the top-scoring documents. At each iteration, this approach uses the RankNet algorithm to re-rank a subset of the results. This splits the problem into smaller and easier tasks and generates a new distribution of the results to be learned by the algorithm. Experimental results show that making the learning algorithm iteratively concentrate on the top-scoring results can improve the accuracy of the top ten documents.


The key challenge we address in this paper is how to work with costs that are everywhere either flat or non-differentiable.

One approach to working with a nonsmooth target cost function would be to search for an optimization function which is a good approximation to the target cost, but which is also smooth. However, the sort required by information retrieval cost functions makes this problematic. Even if the target cost depends on only the top few ranked positions after sorting, the sort itself depends on all documents returned for the query, and that set can be very large; and since the target costs depend on only the rank order and the labels, the target cost functions are either flat or discontinuous in the scores of all the returned documents. We therefore consider a different approach. We illustrate the idea with an example which also demonstrates the perils introduced by a target / optimization cost mismatch. Let the target cost be WTA (Winner Takes All) and let the chosen optimization cost be a smooth approximation to pairwise error.

Suppose that a ranking algorithm A is being trained, and that at some iteration, for a query for which there are only two relevant documents D1 and D2, A gives D1 rank one and D2 rank n. Then on this query, A has WTA cost zero, but a pairwise error cost of n − 2. If the parameters of A are adjusted so that D1 has rank two, and D2 rank three, then the WTA error is now maximized, but the number of pairwise errors has been reduced by n − 4. Now suppose that at the next iteration, D1 is at rank two, and D2 at rank n-1. The change in D1's score that is required to move it to the top position is clearly less (possibly much less) than the change in D2's score required to move it to the top position. Roughly speaking, we would prefer A to spend a little capacity moving D1 up by one position, than have it spend a lot of capacity moving D2 up by n − 1 positions.

If j1 and j2 are the rank indices of D1 and D2 respectively, then instead of pairwise error, we would prefer an optimization cost C that has the property that

$$
\left|\frac{\partial C}{\partial s_{j1}}\right| \leq \left|\frac{\partial C}{\partial s_{j2}}\right|
$$


---

Given two documents $$x_u$$ and $$x_v$$ associated with a training query $$q$$, a target probability $$\bar{P}_{u,v}$$ is constructed based on their ground truth labels. For example, we can define $$\bar{P}_{u,v} = 1$$, if $$y_{u,v} = 1$$, and $$\bar{P}_{u,v} = 0$$ otherwise. Then, the modeled probability $$P_{u,v}$$ is defined based on the difference between the scores of these two documents given by the scoring function, i.e.,

$$
P_{u,v}(f) = \frac{\exp(f(x_u) - f(x_v))}{1 + \exp(f(x_u) - f(x_v))}
$$

Then the cross-entropy between the target probability and the modeled probability is used as the loss function, which we refer to as the cross-entropy loss for short:

$$
L(f; x_u, x_v, y_{u,v}) = -\bar{P}_{u,v} \log(P_{u,v}(f)) - (1 - \bar{P}_{u,v})\log(1 - P_{u,v}(f))
$$

It is not difficult to verify that the cross-entropy loss is an upper bound of the pairwise 0–1 loss, which is defined by

$$
L_{0-1}(f; x_u, x_v, y_{u,v}) =
\begin{cases}
1, & y_{u,v}(f(x_u) - f(x_v)) < 0, \\
0, & \text{otherwise.}
\end{cases}
$$





---

 (typically, we have used neural nets, but we have also implemented RankNet using boosted trees, which we will describe below). RankNet training works as follows. The training data is partitioned by query. 

We then apply the cross-entropy cost function, which penalizes the deviation of the model output probabilities from the desired probabilities.  Then the cost is:




Combining the above two equations gives:


The idea of learning via gradient descent is a key concept that appears throughout this paper. This idea is crucial even when the desired cost doesn't have well-posed gradients and even when the model, such as an ensemble of boosted trees, doesn't have differentiable parameters. To update the model, we must specify the gradient of the cost with respect to the model parameters $$w_k$$, and to do that, we need the gradient of the cost with respect to the model scores $$s_i$$. The gradient descent formulation of boosted trees, such as MART, bypasses the need to compute $$\frac{\partial C}{\partial w_k}$$ by directly modeling $$\frac{\partial C}{\partial s_i}$$.

---
