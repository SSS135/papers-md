# PPG Reloaded: An Empirical Study on What Matters in Phasic Policy Gradient

Kaixin Wang <sup>1</sup> Daquan Zhou <sup>2</sup> Jiashi Feng <sup>2</sup> Shie Mannor 1 3

## Abstract

In model-free reinforcement learning, recent methods based on a phasic policy gradient (PPG) framework have shown impressive improvements in sample efficiency and zero-shot generalization on the challenging Procgen benchmark. In PPG, two design choices are believed to be the key contributing factors to its superior performance over PPO: the high level of value sample reuse and the low frequency of feature distillation. However, through an extensive empirical study, we unveil that *policy regularization* and *data diversity* are what actually matters. In particular, we can achieve the same level of performance with low value sample reuse and frequent feature distillation, as long as the policy regularization strength and data diversity are preserved. In addition, we can maintain the high performance of PPG while reducing the computational cost to a similar level as PPO. Our comprehensive study covers all 16 Procgen games in both sample efficiency and generalization setups. We hope it can advance the understanding of PPG and provide insights for future works.

## 1. Introduction

In recent years, model-free deep reinforcement learning (RL) has achieved great success in multiple domains ranging from video games [\(Badia et al.,](#page-8-0) [2020;](#page-8-0) [Cobbe et al.,](#page-8-1) [2021\)](#page-8-1) to robotics control [\(Haarnoja et al.,](#page-8-2) [2018;](#page-8-2) [OpenAI et al.,](#page-9-0) [2019\)](#page-9-0). One family of high-performing model-free algorithms is the actor-critic methods, such as PPO [\(Schulman et al.,](#page-9-1) [2017\)](#page-9-1), A3C [\(Mnih et al.,](#page-9-2) [2016\)](#page-9-2) and IMPALA [\(Espeholt](#page-8-3) [et al.,](#page-8-3) [2018\)](#page-8-3). These methods learn a policy (*actor*) and a value function (*critic*). In deep RL with image observation (*e.g*., video games), the policy and the value function often

*Proceedings of the* 40 th *International Conference on Machine Learning*, Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright 2023 by the author(s).

share a torso network to encode the observations into latent features. While this benefits sharing features between policy and value, jointly training the policy and value function might incur optimization interference and impose artificial restrictions on sample reuse [\(Cobbe et al.,](#page-8-1) [2021\)](#page-8-1).

Recently, [Cobbe et al.](#page-8-1) [\(2021\)](#page-8-1) proposes a phasic policy gradient (PPG) framework, which circumvents the disadvantages of using a shared encoder while preserving its benefits. The PPG framework is the core behind recent improvements [\(Cobbe et al.,](#page-8-1) [2021;](#page-8-1) [Raileanu & Fergus,](#page-9-3) [2021;](#page-9-3) [Moon](#page-9-4) [et al.,](#page-9-4) [2022\)](#page-9-4) in sample efficiency and zero-shot generalization on the large-scale Procgen benchmark [\(Cobbe et al.,](#page-8-4) [2020\)](#page-8-4). Therefore, it is worthwhile to take a closer look at what makes PPG so effective.

The stark difference between PPG and PPO is that PPG decouples the joint training of policy and value function into two separate phases (a policy phase and a distillation phase, see Section [2.2](#page-1-0) for details). With this decoupling, one can make different algorithmic choices for policy and value learning (Table [1\)](#page-1-1). In existing works [\(Cobbe et al.,](#page-8-1) [2021;](#page-8-1) [Moon et al.,](#page-9-4) [2022\)](#page-9-4), two design choices are believed to be the key factors to the superior performance of PPG:

- *High level of value sample reuse,*
- *Low frequency of feature distillation.*

However, these conclusions are drawn from ablation experiments that do not properly disentangle different factors. For example, in [\(Cobbe et al.,](#page-8-1) [2021\)](#page-8-1), increasing the frequency of feature distillation also shrinks the size of the off-policy buffer, consequently reducing the diversity of the buffered data.

To get a clear understanding of what matters in PPG, we conduct a large-scale empirical study on Procgen. Specifically, we focus on the three aspects in Table [1](#page-1-1) and run ablation experiments with proper control of other (possibly confounding) hyperparameters. We aim to elucidate the core contributing factors in PPG and provide insights for future works built upon it. Our study consists of comprehensive experiments covering all 16 Procgen games in both sample efficiency and generalization setups.

Below is a summary of our main findings:

• The low frequency of feature distillation is actually

<sup>1</sup> Faculty of Electrical And Computer Engineering, Technion, Haifa, Israel <sup>2</sup>ByteDance, Singapore <sup>3</sup>NVIDIA Research, Haifa, Israel. Correspondence to: Kaixin Wang <kaixin96.wang@gmail.com>.

<span id="page-1-1"></span>Table 1. Main differences between the policy training (i.e., the policy phase) and value training (i.e., the distillation phase) in PPG. In comparison, the joint training in PPO restricts the policy and value to be trained with the same (on-policy) data, the same level of sample reuse, and the same update frequency.

| ASPECTS           | $\pi$ (policy phase) | V (distillation phase) |  |
|-------------------|----------------------|------------------------|--|
| DATA SAMPLE REUSE | on-policy<br>low     | off-policy<br>high     |  |
| UPDATE FREQUENCY  | high                 | low                    |  |

not critical. Feature distillation can be performed frequently without degrading the performance, as long as there is sufficiently strong policy regularization.

- High sample reuse is also not the key factor. Reducing the minibatch size while fixing the number of gradient updates (hence lower sample reuse) can achieve similar or even better performance. Besides, we can maintain PPG's high performance while reducing the computational cost to a similar level as PPO.
- Apart from policy regularization, the diversity of training data in the distillation phase is another key contributing factor to the performance.

#### 2. Backgrounds

#### 2.1. Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) (Schulman et al., 2017) is a model-free policy gradient method that optimizes a clipped surrogate objective

$$L^{clip} = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

where  $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$  is the importance sampling ratio to correct the discrepancy between current policy  $\pi_{\theta}$  and the behavior policy  $\pi_{\theta_{old}}$ ,  $\hat{A}_t$  is the estimated advantage at timestep t, and  $\hat{\mathbb{E}}[\ldots]$  indicates the empirical average over a finite batch of timesteps t. An entropy bonus  $H[\pi_{\theta}(\cdot|s)]$  is often used to ensure sufficient exploration. The total loss for training policy  $\pi_{\theta}$  is

$$L_{\pi} = L^{clip} + \beta_H \hat{\mathbb{E}}_t \left[ H[\pi_{\theta}(\cdot|s_t)] \right]$$

where  $\beta_H$  is a scalar hyper-parameter. PPO is commonly implemented in the actor-critic style (Konda & Tsitsiklis, 1999), where a state value function  $V_{\theta}$  is learned for computing variance-reduced advantage. The value function is trained by optimizing

$$L_{V} = \hat{\mathbb{E}}_{t} \left[ \frac{1}{2} \left( V_{\theta} - \hat{V}_{t}^{\text{targ}} \right)^{2} \right]$$

![](_page_1_Picture_14.jpeg)

Figure 1. The network architecture of the single-network PPG.

<span id="page-1-2"></span>where  $\hat{V}_t^{\mathrm{targ}}$  are value function targets. Both  $\hat{A}$  and  $\hat{V}^{\mathrm{targ}}$  are often calculated using Generalized Advantage Estimation (GAE) (Schulman et al., 2016).

In practice, especially when it comes to high-dimensional visual inputs (*e.g.*, video games), the policy and the value function often share a torso network for feature extraction. The state inputs are first mapped to latent features by the torso network, and then the latent features are mapped to policy and value output by separate heads. Empirical evidence on Procgen (Cobbe et al., 2021) suggests that sharing parameters between the policy and the value function leads to clearly better performance than using disjoint networks, probably because this allows features trained by one objective to help better optimize the other.

In the rest of the paper, we denote the parameters of the shared feature encoder, the policy head, and the value head as  $\theta_{\phi}$ ,  $\theta_{\pi}$ , and  $\theta_{V}$  respectively. For notation clarity, we slightly abuse  $\theta$  in the subscript to denote all parameters in the policy and value function, *i.e.*,  $\theta$  in  $V_{\theta}$  refers to  $(\theta_{\phi}, \theta_{V})$  and  $\theta$  in  $\pi_{\theta}$  refers to  $(\theta_{\phi}, \theta_{\pi})$ .

#### <span id="page-1-0"></span>2.2. Phasic Policy Gradient (PPG)

Despite the benefits of feature sharing, the joint training of policy and value in PPO suffers two problems: potential interference between policy and value function optimization and the artificial restriction that policy and value function are trained with the same data. Phasic Policy Gradient (PPG) (Cobbe et al., 2021) addresses these problems by decoupling the training of the policy and value function. While PPG was initially developed to use disjoint policy and value networks, this is not the key to its superior performance, and using the single-network version can achieve comparable performance (see Section 3.6 in (Cobbe et al., 2021)). Therefore, in this paper, we concentrate our focus on the single-network PPG (illustrated in Figure 1).

Different from PPO which jointly trains the policy and value function, PPG decouples its training by alternating between a policy phase and a distillation phase.

In the policy phase, PPG optimizes the following loss:

$$L_{\pi} + \beta_V R_V$$

#### Algorithm 1 Single-network PPG framework

```
1: Initialize a FIFO buffer B of size Toff × Nenv ×M to store off-policy data
 2: for iteration i = 1, 2, . . . do
 3: // Data collection
 4: Roll out policy π to obtain on-policy data D of size Nenv × M
 5: Add on-policy data D to buffer B
 6: // Policy phase
 7: for policy update = 1, 2, . . . , Kπ do
 8: From on-policy data D, sample a minibatch B of size Mπ
 9: Optimize θϕ, θπ and θV with loss Lπ + βV RV on data B
10: end for
11: if t mod Tfreq = 0 then
12: // Distillation phase
13: for value update = 1, 2, . . . , KV do
14: From off-policy buffer B, sample a minibatch B of size MV
15: Optimize θϕ, θπ and θV with loss LV + βπRπ on data B
16: end for
17: end if
                                                                        Toff: buffer size, i.e., how many recent itera-
                                                                        tions of data stored in the buffer
                                                                        Tfreq: number of iterations between two con-
                                                                        secutive distillation phase
                                                                        Nenv: number of parallel environments
                                                                        M: number of roll-out steps
                                                                        Kπ: number of policy updates per iteration
                                                                        KV : number of value updates per iteration
                                                                        βV : value regularization strength
                                                                        βπ: policy regularization strength
```

where R<sup>V</sup> is a regularizer that updates θ<sup>V</sup> while minimizing interference with θϕ. In the original PPG, R<sup>V</sup> is L<sup>V</sup> with gradients detached at the last layer of the shared encoder, such that it only updates θ<sup>V</sup> . Alternatively, [\(Moon et al.,](#page-9-4) [2022\)](#page-9-4) proposes

$$R_V = \hat{\mathbb{E}}_t \left[ \frac{1}{2} \left( V_{\theta}(s_t) - V_{\theta_{\textit{old}}}(s_t) \right)^2 \right],$$

which regularizes the deviation between the updated value function and the old one before the policy phase. In both cases, β<sup>V</sup> is a coefficient to balance the losses.

In the distillation phase, PPG optimizes the following loss:

$$L_V + \beta_\pi R_\pi,$$
 where  $R_\pi = \hat{\mathbb{E}}_t \left[ D_{\text{KL}} \left[ \pi_{\theta_{old}}(\cdot|s_t) \mid \pi_{\theta}(\cdot|s_t) \right] \right]$ 

regularizes the KL divergence between the updated policy and the old policy before the distillation phase. This phase essentially distills useful features learned from the value function objective to the policy. β<sup>π</sup> is a hyperparameter to balance the losses.

Putting it together, PPG decouples the optimization of the policy and the value function: in the policy phase, we optimize L<sup>π</sup> while regularizing V , and in the distillation phase, we optimize L<sup>V</sup> while regularizing π. In each phase, the shared encoder is only updated by either L<sup>π</sup> or L<sup>V</sup> , thus reducing interference.

## 3. Experiments

18: end for

By decoupling the optimization of the policy and value function objectives, PPG is able to train these two objectives differently to boost performance. The differences lie mainly in three aspects. Compared to the policy objective, the value function objective is trained

- much less frequently,
- using additional off-policy data,
- with a higher level of sample reuse.

In comparison, PPO trains both objectives at the same frequency, with the same (on-policy) data, and at the same level of sample reuse.

Among the three points listed above, the lower frequency and the higher sample reuse of the distillation phase (*i.e*., where the value function is trained), are believed to be key to PPG's superior performance [\(Cobbe et al.,](#page-8-1) [2021;](#page-8-1) [Moon](#page-9-4) [et al.,](#page-9-4) [2022\)](#page-9-4). However, the conclusion is drawn from ablation experiments that do not carefully disentangle different factors. For example, in [\(Cobbe et al.,](#page-8-1) [2021\)](#page-8-1), increasing the distillation frequency is coupled with reducing the amount of additional off-policy data. It is hard to tell which is the actual cause of the performance change.

To gain a clear understanding of what matters in PPG, in this section, we carefully control different factors and conduct a large-scale empirical study. Section [3.1](#page-2-0) introduces the experimental settings. Section [3.2,](#page-3-0) [3.3,](#page-4-0) and [3.4](#page-6-0) cover our empirical investigations regarding the three aspects respectively.

#### <span id="page-2-0"></span>3.1. Settings

#### 3.1.1. ALGORITHM FRAMEWORK

The PPG framework used in our experiments is presented in Algorithm [1.](#page-2-1) As introduced in Section [2.2,](#page-1-0) there are

![](_page_3_Figure_1.jpeg)

<span id="page-3-1"></span>Figure 2. Performance with varying distillation interval T*freq* and the default policy regularization strength (β<sup>π</sup> = 1), normalized over all Procgen games.

two ways to regularize the value function during the policy phase (detaching the gradient or penalizing the deviation). In our study, we use the latter since it is shown to perform better [\(Moon et al.,](#page-9-4) [2022\)](#page-9-4). To facilitate later discussions, we make a small change in describing the algorithm: the number of epochs used in prior works is replaced with the number of gradient updates (Lines 7 and 13).

#### 3.1.2. TRAINING AND EVALUATION CONFIGURATIONS

The large-scale Procgen benchmark [\(Cobbe et al.,](#page-8-4) [2020\)](#page-8-4) is used as the testbed of our study. Procgen consists of 16 games that are designed to be highly diverse. Thus, we expect our findings on this benchmark might be useful in other RL environments. In Procgen games, the game level is randomly generated at the beginning of each episode, so the total number of levels is almost infinite. Procgen provides two difficulty choices for each game (*easy* and *hard*), which controls the complexity of the generated levels.

We consider two standard setups in Procgen:

- sample efficiency setup: the agent is trained and tested on the full distribution of levels,
- generalization setup: the agent is trained on a limited set of levels and tested on the full distribution of levels.

Previous works [\(Cobbe et al.,](#page-8-1) [2021;](#page-8-1) [Moon et al.,](#page-9-4) [2022\)](#page-9-4) only focus on one of the above two setups. In comparison, our study covers both, providing a comprehensive view of how different factors contribute to PPG. For the sample efficiency setup, we follow [\(Cobbe et al.,](#page-8-1) [2021\)](#page-8-1) to use the *hard* difficulty and train the agent for 100M steps. For the generalization setup, we consider both *easy* and *hard* difficulties and train the agent for 25M and 200M steps respectively. Other training details can be found in Appendix [A.](#page-10-0) Notably, we reiterate here the default values of some important parameters: T*off* = 32, T*freq* = 32, β<sup>π</sup> = 1. For a fair comparison,

whenever the T*freq* is changed, we also change the number of value updates K<sup>V</sup> accordingly such that the total number of value updates remains the same.

The main performance metric is the mean normalized return averaged over all games in Procgen (see Section 2.2 in [\(Cobbe et al.,](#page-8-4) [2020\)](#page-8-4)). Each experiment is run with 3 random seeds and the standard deviations are plotted as shaded areas. In addition, we also report results using the rliable library [\(Agarwal et al.,](#page-8-5) [2021\)](#page-8-5) in Appendix [B.4.](#page-14-0)

#### <span id="page-3-0"></span>3.2. Distillation Frequency

We first take a closer look at how the frequency of the distillation phase affects performance. In prior works, [Cobbe](#page-8-1) [et al.](#page-8-1) [\(2021\)](#page-8-1) find that performing the distillation phase too frequently has negative impacts. However, their ablation experiments do not control the amount of off-policy data in the buffer B. The buffer size changes as the frequency changes, which may confound the results. In this subsection, we keep the buffer size fixed to eliminate its influence and focus on the interplay between the distillation frequency T*freq* and another overlooked factor: policy regularization strength βπ. The influence of the buffer size will be investigated in the next subsection.

When the features are distilled from the value function to the policy, the policy regularization strength β<sup>π</sup> controls the distortions to the policy between two policy phases. Thus, more frequent distillation leads to more policy distortions throughout policy optimization. We hypothesize that the increased policy distortion is the underlying cause of the performance degradation and that imposing a higher regularization strength can be a simple fix. First, we obtain the results of varying T*freq* but the same regularization strength βπ. As shown in Figure [2,](#page-3-1) with the influence of data diversity removed, the performance seems to still suffer from

![](_page_4_Figure_1.jpeg)

<span id="page-4-1"></span>Figure 3. Performance with varying distillation interval T*freq* and regularization strength βπ, normalized over all Procgen games.

![](_page_4_Figure_3.jpeg)

<span id="page-4-2"></span>Figure 4. Performance with varying regularization strength β<sup>π</sup> and infrequent distillation (T*freq* = 32), normalized over all Procgen games.

more frequent distillation. However, as Figure [3](#page-4-1) shows, if we increase the policy regularization strength as the distillation phase becomes more frequent, we can easily reach a similar performance. In particular, we can perform the policy training phase and distillation phase at the same rate (*i.e*., T*freq* = 1) with almost no performance drop. Thus, we can remove Line 11 in Algorithm [1](#page-2-1) and obtain a simplified PPG. In each iteration, the agent is first trained with policy loss and value regularization and then trained with value loss and policy regularization. We believe removing the infrequency requirement makes PPG easier to analyze since we do not need to consider policy updates across multiple rounds of data collection. On the practical side, this simplification also removes one tunable hyperparameter T*freq*.

Finally, we complete our investigation with experiments that vary the regularization strength under frequent distillation (T*freq* = 32). As shown in Figure [4,](#page-4-2) using higher β<sup>π</sup> leads to similar or even worse performance. The results indicate that high regularization strength works not by improving performance on its own, but rather by fixing the increased policy distortions caused by frequent distillation.

#### <span id="page-4-0"></span>3.3. Data Diversity

As mentioned earlier, in previous work [\(Cobbe et al.,](#page-8-1) [2021\)](#page-8-1), one factor that is entangled with the distillation frequency is the size of the off-policy data buffer. In the above section, we investigate the interplay between distillation frequency and policy regularization while fixing the buffer size to its default value (T*off* = 32). We now turn to this factor and conduct experiments with varying T*off*. A lower T*off* means less diverse data in the buffer. In the extreme case (T*off* = 1), only on-policy data is used for the distillation phase.

First, we fix the policy regularization strength (β<sup>π</sup> = 1) and study how much the performance will be impacted if we reduce T*off* in addition to reducing T*freq*. This experiment helps elucidate if, and how much, the change in data diversity contributes to performance degradation. The results are shown in Figure [5.](#page-5-0) Comparing the solid and dashed lines of the same color and the gray line, we can see that reducing

![](_page_5_Figure_1.jpeg)

<span id="page-5-0"></span>Figure 5. Performance with varying buffer size T*off* at different distillation frequencies, normalized over all Procgen games. Here β<sup>π</sup> is fixed at 1.

![](_page_5_Figure_3.jpeg)

<span id="page-5-1"></span>Figure 6. Performance with varying T*off* under frequent distillation (T*freq* = 1 and β<sup>π</sup> = 16), normalized over all Procgen games.

T*off* indeed leads to a significant performance drop. Please also refer to Figure [19](#page-15-0) to Figure [23](#page-16-0) in the appendix. In addition, this drop loosely correlates to the reduction in T*off*, with changing T*off* from 32 to 1 giving the largest decline.

In Section [3.2,](#page-3-0) we show that it is possible to recover comparable performance under frequent distillation by using a stronger policy regularization (see Figure [3\)](#page-4-1). A natural question is: how does the data diversity affect the results in that case? To answer this question, we run experiments with varying T*off* while setting T*freq* = 1 and β<sup>π</sup> = 16. As the results in Figure [6](#page-5-1) show, using more diverse data (*i.e*., a higher T*off*) generally yields better performance. Only using the on-policy data (*i.e*., T*off* = 1) incurs a considerable performance drop.

In the above results for generalization setups, we observe that the influence of data diversity on the testing performance is less significant compared to that on the training performance. A possible explanation is that, whether T*off* is low or high, the data in the buffer comes from a limited number of training levels. Thus, changes in the data diversity

have a smaller influence on the testing performance.

Finally, we end this subsection with an experiment that varies the data diversity in an alternative way. Specifically, we fix T*off* = 1 (*i.e*., only using on-policy data) but scale the number of parallel environments. To keep the number of samples in each iteration unchanged, we sample a subset of the original size (N*env* ×M) from the collected data and use it instead of all collected data as D for subsequent training. Building upon the results in the previous subsection, we run the experiments under frequent distillation (T*freq* = 1 and β<sup>π</sup> = 16). As the results in Figure [7](#page-6-1) show, despite that only on-policy data is used, increasing the number of environments achieves a similar performance boost as using an off-policy buffer. This indicates that data diversity is indeed one of the factors that matter. We observe that the improvements of the final testing performance in the generalization (hard) setup are relatively small, probably because the default configuration N*env* = 256 is already large enough to provide diverse data.

![](_page_6_Figure_1.jpeg)

<span id="page-6-1"></span>Figure 7. Performance with varying data diversity by scaling the number of parallel environments, normalized over all Procgen games.

![](_page_6_Figure_3.jpeg)

<span id="page-6-2"></span>Figure 8. Performance with scaled minibatch sizes under frequent distillation (T*freq* = 1, β<sup>π</sup> = 16), normalized over all Procgen games.

#### <span id="page-6-0"></span>3.4. Sample Reuse

In this subsection, we look at the last aspect: sample reuse. In the prior work [\(Cobbe et al.,](#page-8-1) [2021\)](#page-8-1), high sample reuse in the distillation phase is attributed as one of the key contributing factors to PPG's good performance. Given a dataset (*e.g*., the off-policy buffer B), two parameters together determine the level of sample reuse: the number of updates (K<sup>V</sup> ) and the minibatch size (M<sup>V</sup> ). Previously, increasing sample reuse is done by increasing the number of updates while keeping the minibatch size fixed. It is possible that the performance boost is the result of more gradient updates rather than higher sample reuse.

To test this hypothesis, we conduct experiments by fixing the number of updates K<sup>V</sup> while varying the minibatch size M<sup>V</sup> . As the results in Figure [8](#page-6-2) show, reducing the minibatch size does not cause a drop in the performance. The only noticeable drop occurs in the generalization (easy) setup, when the minibatch size is too small (M<sup>V</sup> × <sup>1</sup>/<sup>8</sup> = 128). In some cases, especially the sample efficiency setup, a

smaller minibatch size even leads to better results. The results suggest high sample reuse is not a key factor in PPG. We can achieve similar or even better performance with low sample reuse. We also run experiments in the infrequent distillation case (*i.e*., T*freq* = 1 and β<sup>π</sup> = 16), and observe similar results (Figure [9](#page-11-0) in the appendix).

One practical benefit of smaller minibatch size is shorter training time. Although the original PPG outperforms PPO, it also introduces additional training costs (due to more updates in the distillation phase). Our findings suggest that we can achieve the same high performance without sacrificing too much in training speed. Table [2](#page-7-0) shows the average wall clock training time of each iteration under our hardware. We can see that using M<sup>V</sup> × <sup>1</sup>/<sup>8</sup> halves the training time, resulting in similar time costs as PPO. While the numbers in Table [2](#page-7-0) are certainly influenced by various things, we believe the relative gain in training speed is clear.

<span id="page-7-0"></span>Table 2. Comparison of the wall clock training time for each iteration (in seconds), averaged over all runs and games.

|            | PPG with varying MV |      |      |      | PPO  |      |
|------------|---------------------|------|------|------|------|------|
|            | ×2                  | ×1   | ×1/2 | ×1/4 | ×1/8 |      |
| EFF (HARD) | 52.7                | 31.2 | 20.4 | 15.5 | 13.0 | 12.5 |
| GEN (EASY) | 13.3                | 8.5  | 6.2  | 5.0  | 4.6  | 3.9  |
| GEN (HARD) | 51.6                | 31.5 | 20.7 | 15.3 | 13.4 | 12.7 |

## 4. Discussions

Value regularization. In Section [3.2,](#page-3-0) we show that policy regularization is one of the key factors in PPG. Since Algorithm [1](#page-2-1) has a pretty symmetric structure under frequent distillation (*i.e*., removing Line 11), we are interested in how the counterpart of policy regularization, value regularization, affects the performance. Figure [10](#page-11-1) in the appendix shows the results of varying value regularization strength β<sup>V</sup> . As expected, using too strong regularization hampers policy learning and degrades performance.

Interestingly, applying no value regularization (*i.e*., β<sup>V</sup> = 0) only causes little performance drop or even improves testing performance in the generalization (hard) setup. In comparison, the performance will degrade greatly with no value regularization in the infrequent distillation case (Figure [11\)](#page-11-2). The results indicate that we need stronger value regularization under infrequent distillation, which is opposite to our observations for policy regularization. Despite the symmetric structure, policy and value regularization seem to play quite different roles in the training process.

Stiffness and infrequent value update. Stiffness is a measure of how a small gradient step in the network's parameters on one example affects the loss on another example [\(Fort](#page-8-6) [et al.,](#page-8-6) [2019\)](#page-8-6). It has been shown that a value network with lower stiffness is prone to memorizing the training data [\(Ben](#page-8-7)[gio et al.,](#page-8-7) [2020\)](#page-8-7). In the previous work, [Moon et al.](#page-9-4) [\(2022\)](#page-9-4) show that delayed value update (*i.e*., infrequent distillation) leads to higher stiffness and hence helps mitigate overfitting. Since our results indicate that infrequent distillation is not necessary, we are interested in the stiffness measure in such cases. As shown in Figure [12](#page-12-0) (see Appendix [B.3\)](#page-12-1), less frequent distillation in general correlates with higher stiffness. However, compared to Figure [13,](#page-13-0) it seems that the difference in stiffness is not closely correlated with the performance. We believe more works are needed to reach a better understanding of the interactions among stiffness, value update frequency, and overfitting.

## 5. Related Works

PPG is a reinforcement learning framework proposed in [\(Cobbe et al.,](#page-8-1) [2021\)](#page-8-1), aiming to resolve the optimization interference between policy and value function while preserving the benefits of sharing features between them. Following PPG, some works [\(Raileanu & Fergus,](#page-9-3) [2021;](#page-9-3) [Moon et al.,](#page-9-4) [2022\)](#page-9-4) adapt this framework to improve the agent's zero-shot generalization ability. [Raileanu & Fergus](#page-9-3) [\(2021\)](#page-9-3) use the auxiliary head to approximate the advantage instead of the value, in order to reduce the spurious correlation between the observation and the value. [Moon et al.](#page-9-4) [\(2022\)](#page-9-4) introduce explicit value regularization to replace the value network training (with detached gradient) during the policy phase.

While these follow-up works focus on making modifications to PPG, our paper studies what are the key contributing factors in PPG. To the best of our knowledge, the only work that studies different design choices in PPG with comprehensive experiments is the original PPG paper. However, as we show in our paper, the two key factors identified before are actually not that critical. In addition, the previous study only focuses on the sample efficiency setup while our work covers both sample efficiency and generalization setups.

Related to our findings, [Aitchison & Sweetser](#page-8-8) [\(2022\)](#page-8-8) use a distillation loss very similar to the one in PPG and they observe that a smaller batch size works better when optimizing this loss. Apart from Procgen, PPG has also been successfully applied in the very challenging Minecraft environment [\(Baker et al.,](#page-8-9) [2022\)](#page-8-9). Thus, we believe our findings might be useful beyond Procgen games.

## 6. Conclusion and Future Works

In this paper, we investigated what factors matter in PPG, a high-performing actor-critic method. Through comprehensive experiments on the Procgen benchmark, we show that the two factors that were believed to be critical, high level of sample reuse and low frequency of feature distillation, turn out to be not the deciding factors. Instead, we unveil that policy regularization and data diversity are what actually matter in PPG. In addition, our findings suggest that we can preserve PPG's high performance while reducing the computation cost (by more than half) to a similar level as PPO.

We have shown that varying policy regularization strength has a significant influence on the training process (*e.g*., distillation frequency). One interesting topic for future works is to study how policy regularization and value loss influence each other, and how the learned representation evolves. Moreover, what if we use another auxiliary objective other than the value loss? Besides, it is also worthwhile to explore the use of offline data for the distillation phase. Finally, we note that data diversity can be measured and improved consciously, opening up the opportunity for further improvements in the learning process.

## Acknowledgement

This work was partially funded by the Israel Science Foundation under Contract 2199/20.

## References

- <span id="page-8-5"></span>Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., and Bellemare, M. G. Deep reinforcement learning at the edge of the statistical precipice. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), *Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, pp. 29304–29320, 2021. URL [https://proceedings](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).neurips.cc/ [paper/2021/hash/](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) [f514cec81cb148559cf475e7426eed5e-](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)[Abstract](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).html.
- <span id="page-8-8"></span>Aitchison, M. and Sweetser, P. DNA: Proximal policy optimization with a dual network architecture. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), *Advances in Neural Information Processing Systems*, 2022. URL [https://openreview](https://openreview.net/forum?id=WHFgQLRdKf9).net/ [forum?id=WHFgQLRdKf9](https://openreview.net/forum?id=WHFgQLRdKf9).
- <span id="page-8-0"></span>Badia, A. P., Piot, B., Kapturowski, S., Sprechmann, P., Vitvitskyi, A., Guo, Z. D., and Blundell, C. Agent57: Outperforming the atari human benchmark. In *Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pp. 507–517. PMLR, 2020. URL [http://proceedings](http://proceedings.mlr.press/v119/badia20a.html).mlr.press/ [v119/badia20a](http://proceedings.mlr.press/v119/badia20a.html).html.
- <span id="page-8-9"></span>Baker, B., Akkaya, I., Zhokhov, P., Huizinga, J., Tang, J., Ecoffet, A., Houghton, B., Sampedro, R., and Clune, J. Video pretraining (VPT): learning to act by watching unlabeled online videos. *CoRR*, abs/2206.11795, 2022. doi: 10.48550/arXiv.2206.11795. URL [https:](https://doi.org/10.48550/arXiv.2206.11795) //doi.org/10.[48550/arXiv](https://doi.org/10.48550/arXiv.2206.11795).2206.11795.
- <span id="page-8-7"></span>Bengio, E., Pineau, J., and Precup, D. Interference and generalization in temporal difference learning. In *Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pp. 767–777. PMLR, 2020. URL [http://proceedings](http://proceedings.mlr.press/v119/bengio20a.html).mlr.press/ [v119/bengio20a](http://proceedings.mlr.press/v119/bengio20a.html).html.
- <span id="page-8-4"></span>Cobbe, K., Hesse, C., Hilton, J., and Schulman, J. Leveraging procedural generation to benchmark reinforcement learning. In *Proceedings of the 37th International*

- *Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pp. 2048–2056. PMLR, 2020. URL [http://proceedings](http://proceedings.mlr.press/v119/cobbe20a.html).mlr.press/ [v119/cobbe20a](http://proceedings.mlr.press/v119/cobbe20a.html).html.
- <span id="page-8-1"></span>Cobbe, K., Hilton, J., Klimov, O., and Schulman, J. Phasic policy gradient. In Meila, M. and Zhang, T. (eds.), *Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event*, volume 139 of *Proceedings of Machine Learning Research*, pp. 2020–2027. PMLR, 2021. URL [http://proceedings](http://proceedings.mlr.press/v139/cobbe21a.html).mlr.press/ [v139/cobbe21a](http://proceedings.mlr.press/v139/cobbe21a.html).html.
- <span id="page-8-11"></span>Dangel, F., Kunstner, F., and Hennig, P. BackPACK: Packing more into backprop. In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020. URL [https://openreview](https://openreview.net/forum?id=BJlrF24twB).net/ [forum?id=BJlrF24twB](https://openreview.net/forum?id=BJlrF24twB).
- <span id="page-8-3"></span>Espeholt, L., Soyer, H., Munos, R., Simonyan, K., Mnih, V., Ward, T., Doron, Y., Firoiu, V., Harley, T., Dunning, I., Legg, S., and Kavukcuoglu, K. IMPALA: scalable distributed deep-rl with importance weighted actor-learner architectures. In Dy, J. G. and Krause, A. (eds.), *Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmassan, Stockholm, ¨ Sweden, July 10-15, 2018*, volume 80 of *Proceedings of Machine Learning Research*, pp. 1406–1415. PMLR, 2018. URL [http://proceedings](http://proceedings.mlr.press/v80/espeholt18a.html).mlr.press/ [v80/espeholt18a](http://proceedings.mlr.press/v80/espeholt18a.html).html.
- <span id="page-8-6"></span>Fort, S., Nowak, P. K., and Narayanan, S. Stiffness: A new perspective on generalization in neural networks. *CoRR*, abs/1901.09491, 2019. URL [http://arxiv](http://arxiv.org/abs/1901.09491).org/ [abs/1901](http://arxiv.org/abs/1901.09491).09491.
- <span id="page-8-2"></span>Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Dy, J. G. and Krause, A. (eds.), *Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmassan, Stockholm, Swe- ¨ den, July 10-15, 2018*, volume 80 of *Proceedings of Machine Learning Research*, pp. 1856–1865. PMLR, 2018. URL [http://proceedings](http://proceedings.mlr.press/v80/haarnoja18b.html).mlr.press/ [v80/haarnoja18b](http://proceedings.mlr.press/v80/haarnoja18b.html).html.
- <span id="page-8-10"></span>Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In Bengio, Y. and LeCun, Y. (eds.), *3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings*, 2015. URL [http:](http://arxiv.org/abs/1412.6980) //arxiv.[org/abs/1412](http://arxiv.org/abs/1412.6980).6980.

- <span id="page-9-5"></span>Konda, V. R. and Tsitsiklis, J. N. Actor-critic algorithms. In Solla, S. A., Leen, T. K., and Muller, K. ¨ (eds.), *Advances in Neural Information Processing Systems 12, [NIPS Conference, Denver, Colorado, USA, November 29 - December 4, 1999]*, pp. 1008–1014. The MIT Press, 1999. URL [http://papers](http://papers.nips.cc/paper/1786-actor-critic-algorithms).nips.cc/ [paper/1786-actor-critic-algorithms](http://papers.nips.cc/paper/1786-actor-critic-algorithms).
- <span id="page-9-2"></span>Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In Balcan, M. and Weinberger, K. Q. (eds.), *Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016*, volume 48 of *JMLR Workshop and Conference Proceedings*, pp. 1928–1937. JMLR.org, 2016. URL [http://](http://proceedings.mlr.press/v48/mniha16.html) proceedings.mlr.[press/v48/mniha16](http://proceedings.mlr.press/v48/mniha16.html).html.
- <span id="page-9-4"></span>Moon, S., Lee, J., and Song, H. O. Rethinking value function learning for generalization in reinforcement learning. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), *Advances in Neural Information Processing Systems*, 2022. URL [https://openreview](https://openreview.net/forum?id=JkEz1fqN3hX).net/ [forum?id=JkEz1fqN3hX](https://openreview.net/forum?id=JkEz1fqN3hX).
- <span id="page-9-0"></span>OpenAI, Akkaya, I., Andrychowicz, M., Chociej, M., Litwin, M., McGrew, B., Petron, A., Paino, A., Plappert, M., Powell, G., Ribas, R., Schneider, J., Tezak, N., Tworek, J., Welinder, P., Weng, L., Yuan, Q., Zaremba, W., and Zhang, L. Solving rubik's cube with a robot hand. *CoRR*, abs/1910.07113, 2019. URL [http:](http://arxiv.org/abs/1910.07113) //arxiv.[org/abs/1910](http://arxiv.org/abs/1910.07113).07113.
- <span id="page-9-7"></span>Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E. Z., DeVito, ¨ Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: An imperative style, high-performance deep learning library. In Wallach, H. M., Larochelle, H., Beygelzimer, A., d'Alche-Buc, F., Fox, E. B., and Garnett, R. (eds.), ´ *Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada*, pp. 8024–8035, 2019. URL [https://proceedings](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html).neurips.cc/ [paper/2019/hash/](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) [bdbca288fee7f92f2bfa9f7012727740-](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) [Abstract](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html).html.
- <span id="page-9-3"></span>Raileanu, R. and Fergus, R. Decoupling value and policy for generalization in reinforcement learning. In Meila, M. and Zhang, T. (eds.), *Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18- 24 July 2021, Virtual Event*, volume 139 of *Proceedings of Machine Learning Research*, pp. 8787–8798. PMLR,

- 2021. URL [http://proceedings](http://proceedings.mlr.press/v139/raileanu21a.html).mlr.press/ [v139/raileanu21a](http://proceedings.mlr.press/v139/raileanu21a.html).html.
- <span id="page-9-6"></span>Schulman, J., Moritz, P., Levine, S., Jordan, M. I., and Abbeel, P. High-dimensional continuous control using generalized advantage estimation. In Bengio, Y. and Le-Cun, Y. (eds.), *4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings*, 2016. URL [http://arxiv](http://arxiv.org/abs/1506.02438).org/abs/1506.02438.
- <span id="page-9-1"></span>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. *CoRR*, abs/1707.06347, 2017. URL [http:](http://arxiv.org/abs/1707.06347) //arxiv.[org/abs/1707](http://arxiv.org/abs/1707.06347).06347.

## <span id="page-10-0"></span>A. Training Details

For all experiments, we use the residual convolutional networks in IMPALA [\(Espeholt et al.,](#page-8-3) [2018\)](#page-8-3) and the Adam optimizer [\(Kingma & Ba,](#page-8-10) [2015\)](#page-8-10), following previous practices [\(Cobbe et al.,](#page-8-4) [2020;](#page-8-4) [2021\)](#page-8-1). We use PyTorch [\(Paszke et al.,](#page-9-7) [2019\)](#page-9-7) as a deep learning framework. All experiments are conducted using Intel® Xeon® Platinum 8260 CPU and NVIDIA V100 GPU. The table below lists the default values of the parameters in our experiments. Note that when we decrease T*freq*, we also reduce K<sup>V</sup> accordingly such that the total number of gradient updates remains the same. For example, when T*freq* = 1, K<sup>V</sup> = 3072/32 = 96.

|                              | Sample Efficiency (hard) | Generalization (easy) | Generalization (hard) |
|------------------------------|--------------------------|-----------------------|-----------------------|
| Toff                         | 32                       | 32                    | 32                    |
| Tfreq                        | 32                       | 32                    | 32                    |
| Nenv                         | 256                      | 64                    | 256                   |
| M                            | 256                      | 256                   | 256                   |
| Kπ                           | 8                        | 8                     | 8                     |
| KV                           | 3072                     | 3072                  | 3072                  |
| Mπ                           | 8192                     | 2048                  | 8192                  |
| MV                           | 4096                     | 1024                  | 4096                  |
| βV                           | 1                        | 1                     | 1                     |
| βπ                           | 1                        | 1                     | 1                     |
| Discount factor γ            | 0.999                    | 0.999                 | 0.999                 |
| GAE parameter λ              | 0.95                     | 0.95                  | 0.95                  |
| PPO clip range (ϵ)           | 0.2                      | 0.2                   | 0.2                   |
| Reward normalization?        | Yes                      | Yes                   | Yes                   |
| Entropy bonus coefficient βH | 0.01                     | 0.01                  | 0.01                  |
| Learning rate                | 5e-4                     | 5e-4                  | 5e-4                  |
| Maximum gradient norm        | 0.5                      | 0.5                   | 0.5                   |
| Total timesteps              | 100M                     | 25M                   | 200M                  |
| Number of training levels    | All                      | 200                   | 500                   |
| LSTM?                        | No                       | No                    | No                    |
| Frame stack?                 | No                       | No                    | No                    |

## B. More Experiment Results

#### B.1. Scaling minibatch size under infrequent distillation

![](_page_11_Figure_3.jpeg)

<span id="page-11-0"></span>Figure 9. Performance with scaled minibatch sizes under infrequent distillation (T*freq* = 32, β<sup>π</sup> = 1), normalized over all Procgen games.

#### B.2. Varying value regularization strength β<sup>V</sup>

![](_page_11_Figure_6.jpeg)

<span id="page-11-1"></span>Figure 10. Performance with varying β<sup>V</sup> under frequent distillation (T*freq* = 1, β<sup>π</sup> = 16), normalized over all Procgen games.

![](_page_11_Figure_8.jpeg)

<span id="page-11-2"></span>Figure 11. Performance with varying β<sup>V</sup> under infrequent distillation (T*freq* = 32, β<sup>π</sup> = 1), normalized over all Procgen games.

#### <span id="page-12-1"></span>B.3. Stiffness Analysis

Following [\(Moon et al.,](#page-9-4) [2022\)](#page-9-4), we measure the stiffness of the value objective gradients between states. Specifically, the stiffness of the value objective gradients between states (s, s′ ) is defined by

$$\rho(s,s') = \frac{\nabla_{\theta} L_V(s)^{\top} \nabla_{\theta} L_V(s')}{\|\nabla_{\theta} L_V(s)\|_2 \|\nabla_{\theta} L_V(s')\|_2}$$

where θ = (θϕ, θ<sup>V</sup> ) refers to the parameters of the value function, and ∥·∥<sup>2</sup> denotes L2 norm. We run the experiments in the generalization (easy) setup. At each iteration, we first compute the individual value objective gradient for each state in the batch using BackPACK [\(Dangel et al.,](#page-8-11) [2020\)](#page-8-11), then calculate the mean stiffness of the gradients across all state pairs. Finally, the results averaged from 3 runs are shown in the figure below.

![](_page_12_Figure_5.jpeg)

<span id="page-12-0"></span>Figure 12. Stiffness measure under different distillation interval T*freq*.

![](_page_13_Figure_1.jpeg)

<span id="page-13-0"></span>Figure 13. Per-game performance in the generalization (easy) setup, with varying T*freq* and βπ. See Figure [3](#page-4-1) for the normalized result.

#### <span id="page-14-0"></span>B.4. Evaluation results using the rilable library [\(Agarwal et al.,](#page-8-5) [2021\)](#page-8-5)

![](_page_14_Figure_2.jpeg)

Figure 14. Performance in the sample efficiency (hard) setup with varying T*freq* and βπ, corresponding to Figure [2,](#page-3-1) [3](#page-4-1) and [4.](#page-4-2)

![](_page_14_Figure_4.jpeg)

Figure 15. Training performance in the generalization (easy) setup with varying T*freq* and βπ, corresponding to Figure [2,](#page-3-1) [3](#page-4-1) and [4.](#page-4-2)

![](_page_14_Figure_6.jpeg)

Figure 16. Testing performance in the generalization (easy) setup with varying T*freq* and βπ, corresponding to Figure [2,](#page-3-1) [3](#page-4-1) and [4.](#page-4-2)

![](_page_15_Figure_1.jpeg)

Figure 17. Training performance in the generalization (hard) setup with varying T*freq* and βπ, corresponding to Figure [2,](#page-3-1) [3](#page-4-1) and [4.](#page-4-2)

![](_page_15_Figure_3.jpeg)

Figure 18. Testing performance in the generalization (hard) setup with varying T*freq* and βπ, corresponding to Figure [2,](#page-3-1) [3](#page-4-1) and [4.](#page-4-2)

![](_page_15_Figure_5.jpeg)

<span id="page-15-0"></span>Figure 19. Performance in the sample efficiency (hard) setup with varying T*off* at different T*freq*, corresponding to Figure [5.](#page-5-0)

![](_page_16_Figure_1.jpeg)

Figure 20. Training performance in the generalization (easy) setup with varying T*off* at different T*freq*, corresponding to Figure [5.](#page-5-0)

![](_page_16_Figure_3.jpeg)

Figure 21. Testing performance in the generalization (easy) setup with varying T*off* at different T*freq*, corresponding to Figure [5.](#page-5-0)

![](_page_16_Figure_5.jpeg)

Figure 22. Training performance in the generalization (hard) setup with varying T*off* at different T*freq*, corresponding to Figure [5.](#page-5-0)

![](_page_16_Figure_7.jpeg)

<span id="page-16-0"></span>Figure 23. Testing performance in the generalization (hard) setup with varying T*off* at different T*freq*, corresponding to Figure [5.](#page-5-0)

![](_page_17_Figure_1.jpeg)

Figure 24. Performance in the sample efficiency (hard) setup with varying T*off* under T*freq* = 1 and β<sup>π</sup> = 16, corresponding to Figure [6.](#page-5-1)

![](_page_17_Figure_3.jpeg)

Figure 25. Training performance in the generalization (easy) setup with varying T*off* under T*freq* = 1 and β<sup>π</sup> = 16, corresponding to Figure [6.](#page-5-1)

![](_page_17_Figure_5.jpeg)

Figure 26. Testing performance in the generalization (easy) setup varying T*off* under T*freq* = 1 and β<sup>π</sup> = 16, corresponding to Figure [6.](#page-5-1)

![](_page_17_Figure_7.jpeg)

Figure 27. Training performance in the generalization (hard) setup with varying T*off* under T*freq* = 1 and β<sup>π</sup> = 16, corresponding to Figure [6.](#page-5-1)

![](_page_17_Figure_9.jpeg)

Figure 28. Testing performance in the generalization (hard) setup varying T*off* under T*freq* = 1 and β<sup>π</sup> = 16, corresponding to Figure [6.](#page-5-1)

![](_page_18_Figure_1.jpeg)

Figure 29. Performance in the sample efficiency (hard) setup with varying data diversity by scaling the number of parallel environments, corresponding to Figure [7.](#page-6-1)

![](_page_18_Figure_3.jpeg)

Figure 30. Training performance in the generalization (easy) setup with varying data diversity by scaling the number of parallel environments, corresponding to Figure [7.](#page-6-1)

![](_page_18_Figure_5.jpeg)

Figure 31. Testing performance in the generalization (easy) setup varying data diversity by scaling the number of parallel environments, corresponding to Figure [7.](#page-6-1)

![](_page_18_Figure_7.jpeg)

Figure 32. Training performance in the generalization (hard) setup with varying data diversity by scaling the number of parallel environments, corresponding to Figure [7.](#page-6-1)

![](_page_18_Figure_9.jpeg)

Figure 33. Testing performance in the generalization (hard) setup varying data diversity by scaling the number of parallel environments, corresponding to Figure [7.](#page-6-1)

![](_page_19_Figure_1.jpeg)

Figure 34. Performance in the sample efficiency (hard) setup with scaled minibatch sizes under frequent distillation (T*freq* = 1, β<sup>π</sup> = 16), corresponding to Figure [8.](#page-6-2)

![](_page_19_Figure_3.jpeg)

Figure 35. Training performance in the generalization (easy) setup with scaled minibatch sizes under frequent distillation (T*freq* = 1, β<sup>π</sup> = 16), corresponding to Figure [8.](#page-6-2)

![](_page_19_Figure_5.jpeg)

Figure 36. Testing performance in the generalization (easy) setup with scaled minibatch sizes under frequent distillation (T*freq* = 1, β<sup>π</sup> = 16), corresponding to Figure [8.](#page-6-2)

![](_page_19_Figure_7.jpeg)

Figure 37. Training performance in the generalization (hard) setup with scaled minibatch sizes under frequent distillation (T*freq* = 1, β<sup>π</sup> = 16), corresponding to Figure [8.](#page-6-2)

![](_page_19_Figure_9.jpeg)

Figure 38. Testing performance in the generalization (hard) setup with scaled minibatch sizes under frequent distillation (T*freq* = 1, β<sup>π</sup> = 16), corresponding to Figure [8.](#page-6-2)