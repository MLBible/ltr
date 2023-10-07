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
\frac{\partial C}{\partial s_i} = \sigma \left(\frac{1}{2}(1 - S_{ij}) - \frac{1}{1 + e^{\sigma(s_i - s_j)}}\right) = -\frac{\partial C}{\partial s_j} \tag{3}
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

# Limitations

The key challenge we address here is how to work with costs that are everywhere either flat or non-differentiable.

One approach to working with a nonsmooth target cost function would be to search for an optimization function which is a good approximation to the target cost, but which is also smooth. However, the sort required by information retrieval cost functions makes this problematic. We therefore consider a different approach. We illustrate the idea with an example which also demonstrates the perils introduced by a target / optimization cost mismatch. Let the target cost be WTA (Winner Takes All) and let the chosen optimization cost be a smooth approximation to pairwise error.

Suppose that a ranking algorithm $$\mathcal{A}$$ is being trained, and that at some iteration, for a query for which there are only two relevant documents $$D_1$$ and $$D_2$$, $$\mathcal{A}$$ gives $$D_1$$ rank one and $$D_2$$ rank $$n$$. Then on this query, $$\mathcal{A}$$ has WTA cost zero, but a pairwise error cost of $$n − 2$$. If the parameters of $$\mathcal{A}$$ are adjusted so that $$D_1$$ has rank two, and $$D_2$$ rank three, then the WTA error is now maximized, but the number of pairwise errors has been reduced by $$n − 4$$. Now suppose that at the next iteration, $$D_1$$ is at rank two, and $$D_2$$ at rank $$n \gg 1$$. The change in $$D_1$$'s score that is required to move it to the top position is clearly less (possibly much less) than the change in $$D_2$$'s score required to move it to the top position. Roughly speaking, we would prefer $$\mathcal{A}$$ to spend a little capacity moving $$D_1$$ up by one position, than have it spend a lot of capacity moving $$D_2$$ up by $$n − 1$$ positions.

If $$j_1$$ and $$j_2$$ are the rank indices of $$D_1$$ and $$D_2$$ respectively, then instead of pairwise error, we would prefer an optimization cost C that has the property that

$$
\left|\frac{\partial C}{\partial s_{j_1}}\right| \gg \left|\frac{\partial C}{\partial s_{j_2}}\right|
$$

whenever $$j_2 \gg j_1$$.

# Factoring RankNet: Speeding Up RankNet Training

The above leads to a factorization that is the key observation that led to LambdaRank: for a given pair of urls $U_i, U_j$ (again, summations over repeated indices are assumed):

$$
\begin{eqnarray}
\frac{\partial C}{\partial w_k} & = & \frac{\partial C}{\partial s_i} \frac{\partial s_i}{\partial w_k} + \frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k} \\
& = & \frac{\partial C}{\partial s_i} \frac{\partial s_i}{\partial w_k} - \frac{\partial C}{\partial s_i} \frac{\partial s_j}{\partial w_k} \quad \text{[using 3]} \\
& = & \frac{\partial C}{\partial s_i} \left(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k}\right) \\
& = & \lambda_{ij}\left(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k}\right)
\end{eqnarray}
$$

Where we have defined:

$$
\lambda_{ij} \equiv \frac{\partial C}{\partial s_i} = \sigma \left(\frac{1}{2}(1 - S_{ij}) - \frac{1}{1 + e^{\sigma(s_i - s_j)}}\right) \tag{4}
$$

Let $I$ denote the set of pairs of indices $$\{i,j\}$$ for which we desire $U_i$ to be ranked differently from $U_j$ (for a given query). $$I$$ must include each pair just once, so it is convenient to adopt the convention that $I$ contains pairs of indices $$\{i,j\}$$ for which $U_i \rhd U_j$, so that $S_{ij} = 1$. Now summing all the contributions to the update of weight $w_k$ gives:

$$
\begin{eqnarray}
\delta w_k & = & -\eta \sum_{\{i,j\} \in I} \lambda_{ij}\left(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k}\right) \\
& \equiv & -\eta \sum_i \lambda_i \frac{\partial s_i}{\partial w_k}
\end{eqnarray}
$$

Where we have introduced the $\lambda_i$ (one $\lambda_i$ for each url: note that the $\lambda$'s with one subscript are sums of the $\lambda$'s with two). To compute $\lambda_i$ (for url $U_i$), we find all $j$ for which $$\{i,j\} \in I$$ and all $k$ for which $$\{k,i\} \in I$$. For the former, we increment $\lambda_i$ by $\lambda_{ij}$, and for the latter, we decrement $\lambda_i$ by $\lambda_{ki}$. For example, if there were just one pair with $U_1 \rhd U_2$, then $$I = \{\{1,2\}\}$$, and $\lambda_1 = \lambda_{12} = -\lambda_2$. In general, we have:

$$
\lambda_i = \sum_{j:\{i,j\} \in I} \lambda_{ij} - \sum_{j:\{j,i\} \in I} \lambda_{ij}
$$

You can think of the $\lambda$'s as little arrows (or forces), one attached to each (sorted) url, the direction of which indicates the direction we'd like the url to move (to increase relevance), the length of which indicates by how much, and where the $\lambda$ for a given url is computed from all the pairs in which that url is a member.

The algorithm is as follows: instead of backpropagating each pair, first $n$ forward propagations are performed to compute the $s_i$; then for each $i = 1,...,n$ the $\lambda_i \equiv \sum_{j \in P_i} \frac{\partial C(s_i, s_j)}{\partial s_i}$ are computed; then to compute the gradients $\frac{\partial s_i}{\partial w_k}$, $n$ forward propagations are performed, and finally, the $n$ backpropagations are done. 

In the original RankNet implementation true stochastic gradient descent was used i.e. the weights were updated after each pair of urls (with different labels) were examined. The above shows that instead, we can accumulate the $\lambda$'s for each url, summing its contributions from all pairs of urls (where a pair consists of two urls with different labels), and then do the update. This is mini-batch learning, where all the weight updates are first computed for a given query, and then applied, but the speedup results from the way the problem factorizes, not from using mini-batch alone. This led to a very significant speedup in RankNet training (since a weight update is expensive, since e.g. for a neural net model, it requires a backprop). In fact, training time dropped from close to quadratic $O(n^2)$ in the number of urls per query to close to linear $O(n)$. It also laid the groundwork for LambdaRank.

# LambdaRank
LambdaRank has been shown to be a very effective ranking algorithm for optimizing IR (information retrieval) measures. It is a pairwise-based approach that leverages the fact that neural net training needs only the gradients of the cost function, not the function values themselves, and it models those gradients using the sorted positions of the documents for a given query. This bypasses two significant problems, namely that typical IR measures, viewed as functions of the model scores, are either flat or discontinuous everywhere, and that those measures require sorting by score, which itself is a non-differentiable operation. 

Suppose we have only two relevant documents $x_1$ and $x_2$, and their current ranks are 3 and 5. Suppose we are using NDCG@1 as the evaluation measure. Then, it is clear that if we can move either $x_1$ or $x_2$ to the top position of the ranked list, we will achieve the maximum NDCG@1.

It is clearly more convenient to move $x_1$ up since the effort will be much smaller than that for $x_2$. So, we can define (but not compute) the “gradient” with regards to the ranking score of $x_1$ (denoted as $s_1 = f(x_1)$) as larger than that with regards to the ranking score of $x_2$ (denoted as $s_2 = f(x_2)$). In other words, we can consider that there is an underlying implicit loss function $L$ in the optimization process, which suggests:

$$
\frac{\partial L}{\partial s_1} > \frac{\partial L}{\partial s_2}
$$

Experiments have shown that modifying Eq. (4) by simply multiplying by the size of the change in NDCG ($$\vert \Delta_{\text{NDCG}} \vert$$) given by swapping the rank positions of $U_1$ and $U_2$ (while leaving the rank positions of all other urls unchanged) gives very good results. Hence in LambdaRank, we imagine that there is a utility $C$ such that:

$$
\lambda_{ij} = \frac{\partial C(s_i - s_j)}{\partial s_i} = -\frac{\sigma}{1+e^{\sigma(s_i-s_j)}} |\Delta\text{NDCG}|
$$

Since here we want to maximize $C$, the update rule for the weights becomes:

$$
w_k \rightarrow w_k + \eta \frac{\partial C}{\partial w_k}
$$

so that:

$$
\delta C = \frac{\partial C}{\partial w_k} \delta w_k = \eta \left(\frac{\partial C}{\partial w_k}\right)^2 > 0
$$

Thus although Information Retrieval measures, viewed as functions of the model scores, are either flat or discontinuous everywhere, the LambdaRank idea bypasses this problem by computing the gradients after the urls have been sorted by their scores. If you want to optimize some other information retrieval measure, such as MRR or MAP, then LambdaRank can be trivially modified to accomplish this: the only change is that $$\vert \Delta\text{NDCG} \vert $$ above is replaced by the corresponding change in the chosen IR measure.


---

Thus a key intuition behind the $\lambda$-gradient is the observation that NDCG does not treat all pairs equally; the cost depends on the global sorted order as well as on the labels. It is due to these two separate factors that LambdaRank can be applied to any IR metric (by substituting that metric for NDCG), and in fact has been shown to be empirically optimal for several such metrics [8,22] (by “empirically optimal”, we mean that the algorithm finds a local optimum for the cost function, which is by no means obvious, given the indirect route that LambdaRank takes in modeling the cost). This motivates our using the LambdaRank gradients as target gradients in MART. Concretely, the $\lambda$-gradients may be written as:

$$
\lambda_{ij} \equiv S_{ij} \left| \Delta \text{NDCG} \cdot \frac{\partial C_{ij}}{\partial o_{ij}}\right|,
$$

where $o_{ij} \equiv s_i - s_j$ is the difference in ranking scores for a pair of documents in a query (here we are using $s_i$ as a shorthand for $f(x_i)$),

$$
C_{ij} \equiv C(o_{ij}) = s_j - s_i + \log(1+e^{s_i-s_j}),
$$

is the cross-entropy cost applied to the logistic of the difference of the scores, $\Delta \text{NDCG}$ is the NDCG gained by swapping those two documents (after sorting all documents by their current scores), and $$S_{ij} \in \{-1,1\}$$ is plus one if document $i$ is more relevant than document $j$ (has a higher label value) and minus one if document $i$ is less relevant than document $j$ (has a lower label value). Note that

$$
\frac{\partial C_{ij}}{\partial o_{ij}} = \frac{\partial C_{ij}}{\partial s_i} = -\frac{1}{1+e^{o_{ij}}},
$$

and that the overall sign of $\lambda_{ij}$ depends only on the labels of documents $i$ and $j$, and not on their rank position. Each point then sums its $\lambda$-gradients for all pairs $P$ in which it occurs:

$$
\sum \lambda_i = \sum_{j\in P} \lambda_{ij}.
$$

LambdaRank has a physical interpretation in which the documents are point masses and the $\lambda$-gradients are forces on those point masses; the $\lambda$’s generated for any given pair of documents are equal and opposite. A positive lambda indicates a push toward the top rank position and a negative lambda indicates a push toward the lower rank positions.



---


It is convenient to refactor the sum: let $P_i$ be the set of indices $j$ for which $$\{i,j\}$$ is a valid pair, and let $D$ be the set of document indices. Then we can write the first term as:

\begin{equation}
\frac{\partial C}{\partial w_k} = \sum_{i \in D} \frac{\partial s_i}{\partial w_k} \sum_{j \in P_i} \frac{\partial C(s_i, s_j)}{\partial s_i} \tag{7}
\end{equation}

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
