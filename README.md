# Deep Learning & RL Paper Collection

Master list of essential papers. Status: ✅ = converted to MD, ❌ = needs download + conversion.

---

## Deep RL — Foundations

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Playing Atari with Deep Reinforcement Learning (DQN) | Mnih et al. | 2013 | 1312.5602 |
| ❌ | Human-level Control through Deep RL (DQN Nature) | Mnih et al. | 2015 | 1509.06461 |
| ❌ | Deep RL with Double Q-Learning (Double DQN) | van Hasselt et al. | 2015 | 1509.06461 |
| ❌ | Prioritized Experience Replay | Schaul et al. | 2015 | 1511.05952 |
| ❌ | Dueling Network Architectures for Deep RL | Wang et al. | 2015 | 1511.06581 |
| ❌ | A Distributional Perspective on RL (C51) | Bellemare et al. | 2017 | 1707.06887 |
| ❌ | Rainbow: Combining Improvements in Deep RL | Hessel et al. | 2017 | 1710.02298 |
## AlphaGo Family & Game-Playing

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Mastering the Game of Go with Deep NN and Tree Search (AlphaGo) | Silver et al. | 2016 | — (Nature) |
| ❌ | Mastering the Game of Go without Human Knowledge (AlphaGo Zero) | Silver et al. | 2017 | — (Nature) |
| ❌ | Mastering Chess and Shogi by Self-Play (AlphaZero) | Silver et al. | 2017 | 1712.01815 |
| ❌ | Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero) | Schrittwieser et al. | 2020 | 1911.08265 |
| ❌ | Online and Offline RL by Planning with a Learned Model (MuZero Unplugged) | Schrittwieser et al. | 2021 | 2104.06294 |
| ❌ | Learning and Planning in Complex Action Spaces (Sampled MuZero) | Hubert et al. | 2021 | 2104.06303 |
| ❌ | Mastering Atari Games with Limited Data (EfficientZero) | Ye et al. | 2021 | 2111.00210 |
| ❌ | EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data | Wang et al. | 2024 | 2403.00564 |
| ❌ | Thinking Fast and Slow with Deep Learning and Tree Search (ExIt) | Anthony et al. | 2017 | 1705.08439 |
| ❌ | Grandmaster Level in StarCraft II using Multi-Agent RL (AlphaStar) | Vinyals et al. | 2019 | — (Nature) |
| ❌ | Mastering the Game of Stratego with Model-Free MARL (DeepNash) | Perolat et al. | 2022 | 2206.15378 |
| ❌ | OpenSpiel: A Framework for Reinforcement Learning in Games | Lanctot et al. | 2019 | 1908.09453 |

## Deep RL — Policy Gradient & Actor-Critic

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Trust Region Policy Optimization (TRPO) | Schulman et al. | 2015 | 1502.05477 |
| ❌ | High-Dimensional Continuous Control Using GAE | Schulman et al. | 2015 | 1506.02438 |
| ❌ | Continuous Control with Deep RL (DDPG) | Lillicrap et al. | 2015 | 1509.02971 |
| ❌ | Asynchronous Methods for Deep RL (A3C) | Mnih et al. | 2016 | 1602.01783 |
| ✅ | Proximal Policy Optimization Algorithms (PPO) | Schulman et al. | 2017 | 1707.06347 |
| ❌ | Addressing Function Approximation Error in Actor-Critic (TD3) | Fujimoto et al. | 2018 | 1802.09477 |
| ❌ | Soft Actor-Critic (SAC) | Haarnoja et al. | 2018 | 1801.01290 |
| ❌ | Multi-Agent Actor-Critic for Mixed Environments (MADDPG) | Lowe et al. | 2017 | 1706.02275 |

## Deep RL — Scalable & Distributed

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | IMPALA: Scalable Distributed Deep-RL | Espeholt et al. | 2018 | 1802.01561 |
| ❌ | Distributed Prioritized Experience Replay (Ape-X) | Horgan et al. | 2018 | 1803.00933 |
| ❌ | SEED RL: Scalable and Efficient Deep-RL | Espeholt et al. | 2019 | 1910.06591 |

## PPO Variants & Follow-ups

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Emergence of Locomotion Behaviours in Rich Environments (DPPO) | Heess et al. | 2017 | 1707.02286 |
| ❌ | DD-PPO: Learning Near-Perfect PointGoal Navigators | Wijmans et al. | 2019 | 1911.00357 |
| ✅ | Phasic Policy Gradient (PPG) | Cobbe et al. | 2020 | 2009.04416 |
| ❌ | Truly Proximal Policy Optimization | Wang et al. | 2019 | 1903.07940 |
| ❌ | DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization | ByteDance Seed | 2025 | 2503.14476 |
| ❌ | Truncated Proximal Policy Optimization (T-PPO) | — | 2025 | 2506.15050 |

## Deep RL — Model-based

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | World Models | Ha & Schmidhuber | 2018 | 1803.10122 |
| ❌ | Learning Latent Dynamics for Planning from Pixels (PlaNet) | Hafner et al. | 2018 | 1811.04551 |
| ❌ | Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) | Hafner et al. | 2020 | 1912.01603 |
| ❌ | Mastering Diverse Domains through World Models (DreamerV3) | Hafner et al. | 2023 | 2301.04104 |
| ❌ | Imagination-Augmented Agents for Deep RL (I2A) | Racaniere et al. | 2017 | 1707.06203 |
| ❌ | Model-Based RL for Atari (SimPLe) | Kaiser et al. | 2019 | 1903.00374 |
| ❌ | When to Trust Your Model: Model-Based Policy Optimization (MBPO) | Janner et al. | 2019 | 1906.08253 |
| ❌ | Deep RL in a Handful of Trials using Probabilistic Dynamics Models (PETS) | Chua et al. | 2018 | 1805.12114 |
| ❌ | Temporal Difference Learning for Model Predictive Control (TD-MPC) | Hansen et al. | 2022 | 2203.04955 |
| ❌ | Temporal Difference Learning for Model Predictive Control (TD-MPC2) | Hansen et al. | 2024 | 2310.16828 |
| ❌ | MOPO: Model-based Offline Policy Optimization | Yu et al. | 2020 | 2005.13239 |
| ❌ | Mastering Atari with Discrete World Models (DreamerV2) | Hafner et al. | 2020 | 2010.02193 |
| ❌ | Transformers are Sample-Efficient World Models (IRIS) | Micheli et al. | 2022 | 2209.00588 |
| ❌ | Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND) | Alonso et al. | 2024 | 2405.12399 |
| ❌ | Diffusion Models Are Real-Time Game Engines (GameNGen) | Valevski et al. | 2024 | 2408.14837 |
| ❌ | Learning Interactive Real-World Simulators (UniSim) | Yang et al. | 2023 | 2310.06114 |
| ❌ | Genie: Generative Interactive Environments | Bruce et al. | 2024 | 2402.15391 |
| ❌ | DayDreamer: World Models for Physical Robot Learning | Wu et al. | 2022 | 2206.14176 |

## Deep RL — Offline & Generalist

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Conservative Q-Learning for Offline RL (CQL) | Kumar et al. | 2020 | 2006.04779 |
| ❌ | Decision Transformer: RL via Sequence Modeling | Chen et al. | 2021 | 2106.01345 |
| ❌ | Offline RL with Implicit Q-Learning (IQL) | Kostrikov et al. | 2021 | 2110.06169 |
| ❌ | A Generalist Agent (Gato) | Reed et al. | 2022 | 2205.06175 |

## Deep Learning — Foundations

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Gradient-Based Learning Applied to Document Recognition (LeNet) | LeCun et al. | 1998 | — (no arxiv) |
| ❌ | ImageNet Classification with Deep CNNs (AlexNet) | Krizhevsky et al. | 2012 | — (no arxiv) |
| ❌ | Generative Adversarial Nets (GAN) | Goodfellow et al. | 2014 | 1406.2661 |
| ❌ | Auto-Encoding Variational Bayes (VAE) | Kingma & Welling | 2013 | 1312.6114 |
| ❌ | Adam: A Method for Stochastic Optimization | Kingma & Ba | 2014 | 1412.6980 |
| ❌ | Sequence to Sequence Learning with Neural Networks | Sutskever et al. | 2014 | 1409.3215 |
| ❌ | Neural Machine Translation by Jointly Learning to Align and Translate (Attention) | Bahdanau et al. | 2014 | 1409.0473 |
| ❌ | Very Deep Convolutional Networks (VGGNet) | Simonyan & Zisserman | 2014 | 1409.1556 |
| ❌ | Going Deeper with Convolutions (GoogLeNet/Inception) | Szegedy et al. | 2014 | 1409.4842 |
| ❌ | Batch Normalization | Ioffe & Szegedy | 2015 | 1502.03167 |
| ❌ | Deep Residual Learning for Image Recognition (ResNet) | He et al. | 2015 | 1512.03385 |
| ❌ | U-Net: Convolutional Networks for Biomedical Image Segmentation | Ronneberger et al. | 2015 | 1505.04597 |
| ❌ | EfficientNet: Rethinking Model Scaling for CNNs | Tan & Le | 2019 | 1905.11946 |
| ❌ | MobileNets: Efficient CNNs for Mobile Vision Applications | Howard et al. | 2017 | 1704.04861 |
| ❌ | MobileNetV2: Inverted Residuals and Linear Bottlenecks | Sandler et al. | 2018 | 1801.04381 |
| ✅ | Layer Normalization | Ba et al. | 2016 | 1607.06450 |
| ✅ | Group Normalization | Wu & He | 2018 | 1803.08494 |
| ❌ | Root Mean Square Layer Normalization (RMSNorm) | Zhang & Sennrich | 2019 | 1910.07467 |
| ✅ | Distilling the Knowledge in a Neural Network | Hinton et al. | 2015 | 1503.02531 |
| ❌ | Deep Networks with Stochastic Depth | Huang et al. | 2016 | 1603.09382 |

## Optimization & Training Techniques

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Decoupled Weight Decay Regularization (AdamW) | Loshchilov & Hutter | 2019 | 1711.05101 |
| ❌ | SGDR: Stochastic Gradient Descent with Warm Restarts | Loshchilov & Hutter | 2017 | 1608.03983 |
| ❌ | mixup: Beyond Empirical Risk Minimization | Zhang et al. | 2018 | 1710.09412 |
| ❌ | CutMix: Regularization Strategy to Train Strong Classifiers | Yun et al. | 2019 | 1905.04899 |
| ❌ | AutoAugment: Learning Augmentation Policies from Data | Cubuk et al. | 2019 | 1805.09501 |
| ❌ | RandAugment: Practical Automated Data Augmentation | Cubuk et al. | 2020 | 1909.13719 |
| ❌ | Neural Architecture Search with Reinforcement Learning | Zoph & Le | 2017 | 1611.01578 |
| ❌ | DARTS: Differentiable Architecture Search | Liu et al. | 2019 | 1806.09055 |

## Pre-Transformer NLP & Embeddings

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Efficient Estimation of Word Representations in Vector Space (word2vec) | Mikolov et al. | 2013 | 1301.3781 |
| ❌ | Deep Contextualized Word Representations (ELMo) | Peters et al. | 2018 | 1802.05365 |
| ❌ | XLNet: Generalized Autoregressive Pretraining for Language Understanding | Yang et al. | 2019 | 1906.08237 |
| ❌ | RoBERTa: A Robustly Optimized BERT Pretraining Approach | Liu et al. | 2019 | 1907.11692 |

## Transformers & NLP

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Attention Is All You Need (Transformer) | Vaswani et al. | 2017 | 1706.03762 |
| ❌ | BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al. | 2018 | 1810.04805 |
| ❌ | Exploring the Limits of Transfer Learning (T5) | Raffel et al. | 2019 | 1910.10683 |
| ❌ | Language Models are Few-Shot Learners (GPT-3) | Brown et al. | 2020 | 2005.14165 |
| ❌ | Scaling Laws for Neural Language Models | Kaplan et al. | 2020 | 2001.08361 |
| ❌ | Training Compute-Optimal Large Language Models (Chinchilla) | Hoffmann et al. | 2022 | 2203.15556 |
| ❌ | LLaMA: Open and Efficient Foundation Language Models | Touvron et al. | 2023 | 2302.13971 |
| ❌ | Mixtral of Experts | Jiang et al. | 2024 | 2401.04088 |
| ✅ | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu & Dao | 2023 | 2312.00752 |
| ✅ | Transformers are SSMs (Mamba-2) | Dao & Gu | 2024 | 2405.21060 |
| ❌ | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG) | Lewis et al. | 2020 | 2005.11401 |

## Efficient Training & Inference

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | FlashAttention: Fast and Memory-Efficient Exact Attention | Dao et al. | 2022 | 2205.14135 |
| ❌ | LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2021 | 2106.09685 |
| ❌ | RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE) | Su et al. | 2021 | 2104.09864 |

## Multimodal & Generative

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Learning Transferable Visual Models from Natural Language Supervision (CLIP) | Radford et al. | 2021 | 2103.00020 |
| ❌ | Denoising Diffusion Probabilistic Models (DDPM) | Ho et al. | 2020 | 2006.11239 |
| ❌ | High-Resolution Image Synthesis with Latent Diffusion Models | Rombach et al. | 2022 | 2112.10752 |
| ❌ | Flamingo: a Visual Language Model for Few-Shot Learning | Alayrac et al. | 2022 | 2204.14198 |
| ❌ | A Style-Based Generator Architecture for GANs (StyleGAN) | Karras et al. | 2019 | 1812.04948 |
| ❌ | Analyzing and Improving the Image Quality of StyleGAN (StyleGAN2) | Karras et al. | 2020 | 1912.04958 |
| ❌ | Zero-Shot Text-to-Image Generation (DALL-E) | Ramesh et al. | 2021 | 2102.12092 |
| ❌ | Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2) | Ramesh et al. | 2022 | 2204.06125 |
| ❌ | Photorealistic Text-to-Image Diffusion Models (Imagen) | Saharia et al. | 2022 | 2205.11487 |

## Vision & Self-Supervised Learning

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | An Image is Worth 16x16 Words (ViT) | Dosovitskiy et al. | 2020 | 2010.11929 |
| ❌ | Training Data-Efficient Image Transformers (DeiT) | Touvron et al. | 2021 | 2012.12877 |
| ❌ | Swin Transformer: Hierarchical Vision Transformer | Liu et al. | 2021 | 2103.14030 |
| ❌ | A Simple Framework for Contrastive Learning (SimCLR) | Chen et al. | 2020 | 2002.05709 |
| ❌ | Bootstrap Your Own Latent (BYOL) | Grill et al. | 2020 | 2006.07733 |
| ❌ | Masked Autoencoders Are Scalable Vision Learners (MAE) | He et al. | 2021 | 2111.06377 |
| ❌ | Emerging Properties in Self-Supervised Vision Transformers (DINO) | Caron et al. | 2021 | 2104.14294 |
| ❌ | DINOv2: Learning Robust Visual Features without Supervision | Oquab et al. | 2023 | 2304.07193 |
| ❌ | Segment Anything (SAM) | Kirillov et al. | 2023 | 2304.02643 |
| ✅ | Representation Learning with Contrastive Predictive Coding (CPC) | van den Oord et al. | 2018 | 1807.03748 |

## Object Detection & Segmentation

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Faster R-CNN: Towards Real-Time Object Detection | Ren et al. | 2015 | 1506.01497 |
| ❌ | Feature Pyramid Networks for Object Detection (FPN) | Lin et al. | 2017 | 1612.03144 |
| ❌ | Mask R-CNN | He et al. | 2017 | 1703.06870 |
| ❌ | YOLOv3: An Incremental Improvement | Redmon & Farhadi | 2018 | 1804.02767 |
| ❌ | End-to-End Object Detection with Transformers (DETR) | Carion et al. | 2020 | 2005.12872 |

## 3D Vision

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | NeRF: Representing Scenes as Neural Radiance Fields | Mildenhall et al. | 2020 | 2003.08934 |
| ❌ | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Kerbl et al. | 2023 | 2308.04079 |
| ❌ | Instant Neural Graphics Primitives with a Multiresolution Hash Encoding | Müller et al. | 2022 | 2201.05989 |

## Graph Neural Networks

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Semi-Supervised Classification with Graph Convolutional Networks (GCN) | Kipf & Welling | 2017 | 1609.02907 |
| ❌ | Graph Attention Networks (GAT) | Velickovic et al. | 2018 | 1710.10903 |
| ❌ | Neural Message Passing for Quantum Chemistry (MPNN) | Gilmer et al. | 2017 | 1704.01212 |

## JEPA Family

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ✅ | Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA) | Assran et al. | 2023 | 2301.08243 |
| ❌ | V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video | Bardes et al. | 2024 | 2404.08471 |
| ❌ | MC-JEPA: Joint-Embedding Predictive Architecture for Motion and Content | Bardes et al. | 2023 | 2307.12698 |
| ❌ | A-JEPA: Joint-Embedding Predictive Architecture Can Listen | Bardes et al. | 2023 | 2311.15830 |
| ✅ | LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics | — | 2025 | 2511.08544 |
| ✅ | JEPA as a Neural Tokenizer: Learning Robust Speech Representations | — | 2025 | 2512.07168 |
| ✅ | Causal-JEPA: Learning World Models through Object-Level Latent Interventions | — | 2025 | 2602.11389 |
| ✅ | LeWorldModel: Stable End-to-End JEPA from Pixels | — | 2025 | 2603.19312 |
| ❌ | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | Assran, Bardes et al. | 2025 | 2506.09985 |
| ❌ | VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language | Chen et al. | 2025 | 2512.10942 |
| ❌ | Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning | Tuncay et al. | 2025 | 2507.02915 |
| ❌ | Point-JEPA: Self-Supervised Learning on Point Cloud | Saito et al. | 2024 | 2404.16432 |
| ❌ | 3D-JEPA: 3D Self-Supervised Representation Learning | Hu et al. | 2024 | 2409.15803 |
| ❌ | T-JEPA: Joint-Embedding Predictive Architecture for Trajectory Similarity | Li et al. | 2024 | 2406.12913 |
| ❌ | S-JEPA: Seamless Cross-Dataset Transfer through Dynamic Spatial Attention (EEG) | Guetschel et al. | 2024 | 2403.11772 |
| ❌ | ECG-JEPA: Self-Supervised Pre-Training Boosts ECG Classification | Weimann & Conrad | 2024 | 2410.13867 |

## LLM-RL & Alignment

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Training Language Models to Follow Instructions with Human Feedback (InstructGPT) | Ouyang et al. | 2022 | 2203.02155 |
| ❌ | Constitutional AI: Harmlessness from AI Feedback | Bai et al. | 2022 | 2212.08073 |
| ❌ | Direct Preference Optimization (DPO) | Rafailov et al. | 2023 | 2305.18290 |
| ❌ | Let's Verify Step by Step (Process Reward Models) | Lightman et al. | 2023 | 2305.20050 |
| ❌ | A General Theoretical Paradigm to Understand Learning from Human Feedback (IPO) | Azar et al. | 2023 | 2310.12036 |
| ❌ | STaR: Bootstrapping Reasoning with Reasoning | Zelikman et al. | 2022 | 2203.14465 |
| ❌ | Self-Play Fine-Tuning Converts Weak LMs to Strong LMs (SPIN) | Chen et al. | 2024 | 2401.01335 |
| ❌ | DeepSeekMath / GRPO: Pushing the Limits of Mathematical Reasoning | DeepSeek-AI | 2024 | 2402.03300 |
| ❌ | KTO: Model Alignment as Prospect Theoretic Optimization | Ethayarajh et al. | 2024 | 2402.01306 |
| ❌ | ORPO: Monolithic Preference Optimization without Reference Model | Hong et al. | 2024 | 2403.07691 |
| ❌ | SimPO: Simple Preference Optimization with a Reference-Free Reward | Meng et al. | 2024 | 2405.14734 |
| ❌ | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Wei et al. | 2022 | 2201.11903 |
| ❌ | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | DeepSeek-AI | 2025 | 2501.12948 |
| ❌ | Scaling LLM Test-Time Compute Optimally | Snell et al. | 2024 | 2408.03314 |

## Speech & Audio

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | WaveNet: A Generative Model for Raw Audio | van den Oord et al. | 2016 | 1609.03499 |
| ❌ | wav2vec 2.0: Self-Supervised Learning of Speech Representations | Baevski et al. | 2020 | 2006.11477 |
| ❌ | Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | Radford et al. | 2022 | 2212.04356 |

## Robotics

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | RT-1: Robotics Transformer for Real-World Control at Scale | Brohan et al. | 2022 | 2212.06817 |
| ❌ | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | Brohan et al. | 2023 | 2307.15818 |

## Science & Theory

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Highly Accurate Protein Structure Prediction with AlphaFold | Jumper et al. | 2021 | — (Nature) |
| ❌ | Neural Ordinary Differential Equations | Chen et al. | 2018 | 1806.07366 |
| ❌ | The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | Frankle & Carlin | 2019 | 1803.03635 |
| ❌ | Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets | Power et al. | 2022 | 2201.02177 |
| ❌ | KAN: Kolmogorov-Arnold Networks | Liu et al. | 2024 | 2404.19756 |

## Frontier LLM Technical Reports

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | GPT-4 Technical Report | OpenAI | 2023 | 2303.08774 |
| ❌ | Gemini: A Family of Highly Capable Multimodal Models | Google DeepMind | 2023 | 2312.11805 |
| ❌ | Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens | Reid et al. | 2024 | 2403.05530 |
| ❌ | The Llama 3 Herd of Models | Meta AI | 2024 | 2407.21783 |
| ❌ | Mistral 7B | Jiang et al. | 2023 | 2310.06825 |
| ❌ | DeepSeek-V2: A Strong, Economical MoE Language Model | DeepSeek-AI | 2024 | 2405.04434 |
| ❌ | DeepSeek-V3 Technical Report | DeepSeek-AI | 2024 | 2412.19437 |
| ❌ | Phi-3 Technical Report | Abdin et al. | 2024 | 2404.14219 |
| ❌ | Phi-4 Technical Report | Abdin et al. | 2024 | 2412.08905 |
| ❌ | Qwen2 Technical Report | Qwen Team | 2024 | 2407.10671 |
| ❌ | Qwen2.5 Technical Report | Qwen Team | 2024 | 2412.15115 |
| ❌ | Gemma 2: Improving Open Language Models at a Practical Size | Google DeepMind | 2024 | 2408.00118 |
| ❌ | OLMo: Accelerating the Science of Language Models | Allen AI | 2024 | 2402.00838 |

## Novel Architectures (2024-2025)

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Eagle and Finch: RWKV with Matrix-Valued States (RWKV-5/6) | Peng et al. | 2024 | 2404.05892 |
| ❌ | Jamba: A Hybrid Transformer-Mamba Language Model | Lieber et al. | 2024 | 2403.19887 |
| ❌ | Byte Latent Transformer: Patches Scale Better Than Tokens | Pagnoni et al. | 2024 | 2412.09871 |
| ❌ | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | Shah et al. | 2024 | 2407.08608 |

## RL Environments & Benchmarks

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | The Arcade Learning Environment: An Evaluation Platform for General Agents (ALE) | Bellemare et al. | 2013 | 1207.4708 |
| ❌ | OpenAI Gym | Brockman et al. | 2016 | 1606.01540 |
| ❌ | Gymnasium: A Standard Interface for RL Environments | Towers et al. | 2024 | 2407.17032 |
| ❌ | MuJoCo: A Physics Engine for Model-Based Control | Todorov et al. | 2012 | — (IROS) |
| ❌ | DeepMind Lab | Beattie et al. | 2016 | 1612.03801 |
| ❌ | DeepMind Control Suite | Tassa et al. | 2018 | 1801.00690 |
| ❌ | Leveraging Procedural Generation to Benchmark RL (Procgen) | Cobbe et al. | 2020 | 1912.01588 |
| ❌ | BabyAI: A Platform to Study Sample Efficiency of Grounded Language Learning | Chevalier-Boisvert et al. | 2019 | 1810.08272 |
| ❌ | Minigrid & Miniworld: Modular RL Environments for Goal-Oriented Tasks | Chevalier-Boisvert et al. | 2023 | 2306.13831 |
| ❌ | The NetHack Learning Environment | Küttler et al. | 2020 | 2006.13760 |
| ❌ | The StarCraft Multi-Agent Challenge (SMAC) | Samvelyan et al. | 2019 | 1902.04043 |
| ❌ | Isaac Gym: High Performance GPU-Based Physics Simulation for Robot Learning | Makoviychuk et al. | 2021 | 2108.10470 |
| ❌ | Brax: A Differentiable Physics Engine for Large Scale Rigid Body Simulation | Freeman et al. | 2021 | 2106.13281 |
| ❌ | PettingZoo: Gym for Multi-Agent Reinforcement Learning | Terry et al. | 2021 | 2009.14471 |
| ❌ | Open-Ended Learning Leads to Generally Capable Agents (XLand) | DeepMind | 2021 | 2107.12808 |
| ❌ | Benchmarking the Spectrum of Agent Capabilities (Crafter) | Hafner | 2021 | 2109.06780 |
| ❌ | MinAtar: An Atari-Inspired Testbed for Thorough RL Experiments | Young & Tian | 2019 | 1903.03176 |
| ❌ | EnvPool: A Highly Parallel RL Environment Execution Engine | Weng et al. | 2022 | 2206.10558 |
| ❌ | Pgx: Hardware-Accelerated Parallel Game Simulators for RL | Koyamada et al. | 2023 | 2303.17503 |
| ❌ | Jumanji: A Diverse Suite of Scalable RL Environments in JAX | InstaDeep | 2023 | 2306.09884 |
| ❌ | JaxMARL: Multi-Agent RL Environments and Algorithms in JAX | Rutherford et al. | 2023 | 2311.10090 |
| ❌ | Craftax: A Lightning-Fast Benchmark for Open-Ended RL | Matthews et al. | 2024 | 2402.16801 |

## Optimizers (beyond Adam/AdamW)

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | ADADELTA: An Adaptive Learning Rate Method | Zeiler | 2012 | 1212.5701 |
| ❌ | Large Batch Training of Convolutional Networks (LARS) | You et al. | 2017 | 1708.03888 |
| ❌ | Large Batch Optimization for Deep Learning: Training BERT in 76 min (LAMB) | You et al. | 2019 | 1904.00962 |
| ❌ | Shampoo: Preconditioned Stochastic Tensor Optimization | Gupta et al. | 2018 | 1802.09568 |
| ❌ | Adafactor: Adaptive Learning Rates with Sublinear Memory Cost | Shazeer & Stern | 2018 | 1804.04235 |
| ❌ | Symbolic Discovery of Optimization Algorithms (Lion) | Chen et al. | 2023 | 2302.06675 |
| ❌ | Sophia: A Scalable Stochastic Second-order Optimizer for LM Pre-training | Liu et al. | 2023 | 2305.14342 |
| ❌ | Learning-Rate-Free Learning by D-Adaptation | Defazio & Mishchenko | 2023 | 2301.07733 |
| ❌ | Prodigy: An Expeditiously Adaptive Parameter-Free Learner | Mishchenko & Defazio | 2023 | 2306.06101 |
| ❌ | The Road Less Scheduled (Schedule-Free) | Defazio et al. | 2024 | 2405.15682 |
| ❌ | GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection | Zhao et al. | 2024 | 2403.03507 |
| ❌ | SOAP: Improving and Stabilizing Shampoo using Adam | Vyas et al. | 2024 | 2409.11321 |

## Activation Functions

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Delving Deep into Rectifiers / PReLU (He Initialization) | He et al. | 2015 | 1502.01852 |
| ❌ | Fast and Accurate Deep Network Learning by ELUs | Clevert et al. | 2015 | 1511.07289 |
| ❌ | Self-Normalizing Neural Networks (SELU) | Klambauer et al. | 2017 | 1706.02515 |
| ❌ | Gaussian Error Linear Units (GELU) | Hendrycks & Gimpel | 2016 | 1606.08415 |
| ❌ | Searching for Activation Functions (Swish/SiLU) | Ramachandran et al. | 2017 | 1710.05941 |
| ❌ | Mish: A Self Regularized Non-Monotonic Activation Function | Misra | 2019 | 1908.08681 |
| ❌ | Language Modeling with Gated Convolutional Networks (GLU) | Dauphin et al. | 2016 | 1612.08083 |
| ❌ | GLU Variants Improve Transformer (SwiGLU) | Shazeer | 2020 | 2002.05202 |

## Normalization & Initialization (additional)

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Instance Normalization: The Missing Ingredient for Fast Stylization | Ulyanov et al. | 2016 | 1607.08022 |
| ❌ | Weight Normalization: A Simple Reparameterization | Salimans & Kingma | 2016 | 1602.07868 |
| ❌ | Spectral Normalization for Generative Adversarial Networks | Miyato et al. | 2018 | 1802.05957 |
| ❌ | On Layer Normalization in the Transformer Architecture (Pre-LN) | Xiong et al. | 2020 | 2002.04745 |
| ❌ | DeepNet: Scaling Transformers to 1000 Layers (DeepNorm) | Wang et al. | 2022 | 2203.00555 |
| ❌ | Fixup Initialization: Residual Learning Without Normalization | Zhang et al. | 2019 | 1901.09321 |
| ❌ | Tensor Programs V: Tuning Large NNs via Zero-Shot Hyperparameter Transfer (muP) | Yang et al. | 2022 | 2203.03466 |

## Regularization (additional)

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | DropBlock: A Regularization Method for Convolutional Networks | Ghiasi et al. | 2018 | 1810.12890 |
| ❌ | Rethinking the Inception Architecture (Label Smoothing) | Szegedy et al. | 2016 | 1512.00567 |
| ❌ | When Does Label Smoothing Help? | Müller et al. | 2019 | 1906.02629 |
| ❌ | Improved Regularization of CNNs with Cutout | DeVries & Taylor | 2017 | 1708.04552 |
| ❌ | Averaging Weights Leads to Wider Optima and Better Generalization (SWA) | Izmailov et al. | 2018 | 1803.05407 |

## RL — Famous Game-Playing Results

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five) | Berner et al. | 2019 | 1912.06680 |
| ❌ | Human-level Performance in First-Person Multiplayer Games (Capture the Flag) | Jaderberg et al. | 2019 | 1807.01281 |
| ❌ | Agent57: Outperforming the Atari Human Benchmark | Badia et al. | 2020 | 2003.13350 |
| ❌ | First Return, Then Explore (Go-Explore) | Ecoffet et al. | 2021 | 2004.12919 |
| ❌ | Video PreTraining: Learning to Act by Watching Unlabeled Online Videos (VPT) | Baker et al. | 2022 | 2206.11795 |
| ❌ | MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge | Fan et al. | 2022 | 2206.08853 |
| ❌ | Voyager: An Open-Ended Embodied Agent with Large Language Models | Wang et al. | 2023 | 2305.16291 |
| ❌ | Competition-Level Code Generation with AlphaCode | Li et al. | 2022 | 2203.07814 |
| ❌ | Solving Rubik's Cube with a Robot Hand (Dactyl) | OpenAI | 2019 | 1910.07113 |

## RL — Practical Tips, Tricks & Engineering

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Deep Reinforcement Learning that Matters | Henderson et al. | 2018 | 1709.06560 |
| ❌ | Implementation Matters in Deep Policy Gradients | Engstrom et al. | 2020 | 2005.12729 |
| ❌ | What Matters In On-Policy Reinforcement Learning? | Andrychowicz et al. | 2021 | 2006.05990 |
| ❌ | CleanRL: High-quality Single-file Implementations of Deep RL Algorithms | Huang et al. | 2022 | 2111.08819 |
| ❌ | Hindsight Experience Replay (HER) | Andrychowicz et al. | 2017 | 1707.01495 |
| ❌ | Curiosity-driven Exploration by Self-supervised Prediction (ICM) | Pathak et al. | 2017 | 1705.05363 |
| ❌ | Exploration by Random Network Distillation (RND) | Burda et al. | 2019 | 1810.12894 |
| ❌ | Unifying Count-Based Exploration and Intrinsic Motivation | Bellemare et al. | 2016 | 1606.01868 |
| ❌ | Count-Based Exploration with Neural Density Models | Ostrovski et al. | 2017 | 1703.01310 |
| ❌ | VIME: Variational Information Maximizing Exploration | Houthooft et al. | 2016 | 1605.09674 |
| ❌ | Never Give Up: Learning Directed Exploration Strategies (NGU) | Badia et al. | 2020 | 2002.06038 |
| ❌ | Noisy Networks for Exploration (NoisyNet) | Fortunato et al. | 2017 | 1706.10295 |
| ❌ | Deep Exploration via Bootstrapped DQN | Osband et al. | 2016 | 1602.04621 |
| ❌ | Planning to Explore via Self-Supervised World Models (Plan2Explore) | Sekar et al. | 2020 | 2005.05960 |
| ❌ | RIDE: Rewarding Impact-Driven Exploration | Raileanu & Rocktäschel | 2020 | 2002.12292 |
| ❌ | URLB: Unsupervised Reinforcement Learning Benchmark | Laskin et al. | 2021 | 2110.15191 |
| ❌ | Diversity is All You Need: Learning Skills without a Reward Function (DIAYN) | Eysenbach et al. | 2018 | 1802.06070 |

## Multi-Agent RL & Self-Play

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Population Based Training of Neural Networks (PBT) | Jaderberg et al. | 2017 | 1711.09846 |
| ❌ | QMIX: Monotonic Value Function Factorisation for Deep MARL | Rashid et al. | 2018 | 1803.11485 |
| ❌ | The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO) | Yu et al. | 2021 | 2103.01955 |
| ❌ | Deep RL from Self-Play in Imperfect-Information Games (NFSP) | Heinrich & Silver | 2016 | 1603.01121 |
| ❌ | Learning with Opponent-Learning Awareness (LOLA) | Foerster et al. | 2017 | 1709.04326 |
| ❌ | Mean Field Multi-Agent Reinforcement Learning | Yang et al. | 2018 | 1802.05438 |
| ❌ | Counterfactual Multi-Agent Policy Gradients (COMA) | Foerster et al. | 2017 | 1705.08926 |
| ❌ | Value-Decomposition Networks for Cooperative MARL (VDN) | Sunehag et al. | 2017 | 1706.05296 |
| ❌ | A Unified Game-Theoretic Approach to Multiagent RL (PSRO) | Lanctot et al. | 2017 | 1711.00832 |
| ❌ | Emergent Complexity via Multi-Agent Competition | Bansal et al. | 2017 | 1710.03748 |
| ❌ | RODE: Learning Roles to Decompose Multi-Agent Tasks | Wang et al. | 2021 | 2010.01523 |
| ❌ | Weighted QMIX: Expanding Monotonic Value Function Factorisation | Rashid et al. | 2020 | 2006.10800 |
| ❌ | MAVEN: Multi-Agent Variational Exploration | Mahajan et al. | 2019 | 1910.07483 |
| ❌ | Learning Multiagent Communication with Backpropagation (CommNet) | Sukhbaatar et al. | 2016 | 1605.07736 |
| ❌ | Learning to Communicate with Deep Multi-Agent RL (DIAL/RIAL) | Foerster et al. | 2016 | 1605.06676 |
| ❌ | TarMAC: Targeted Multi-Agent Communication | Das et al. | 2019 | 1810.11187 |
| ❌ | The Hanabi Challenge: A New Frontier for AI Research | Bard et al. | 2019 | 1902.00506 |
| ❌ | On the Utility of Learning about Humans for Human-AI Coordination (Overcooked-AI) | Carroll et al. | 2019 | 1910.05789 |
| ❌ | Scalable Evaluation of MARL with Melting Pot | Leibo et al. | 2021 | 2107.06857 |
| ❌ | SMACv2: An Improved Benchmark for Cooperative MARL | Ellis et al. | 2022 | 2212.07489 |

## Adversarial Training & Robustness

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Intriguing Properties of Neural Networks | Szegedy et al. | 2013 | 1312.6199 |
| ❌ | Explaining and Harnessing Adversarial Examples (FGSM) | Goodfellow et al. | 2014 | 1412.6572 |
| ❌ | Towards Deep Learning Models Resistant to Adversarial Attacks (PGD) | Madry et al. | 2017 | 1706.06083 |
| ❌ | Adversarial Training for Free! | Shafahi et al. | 2019 | 1904.12843 |
| ❌ | Theoretically Principled Trade-off between Robustness and Accuracy (TRADES) | Zhang et al. | 2019 | 1901.08573 |
| ❌ | Virtual Adversarial Training (VAT) | Miyato et al. | 2017 | 1704.03976 |
| ❌ | Certified Adversarial Robustness via Randomized Smoothing | Cohen et al. | 2019 | 1902.02918 |
| ❌ | Obfuscated Gradients Give a False Sense of Security | Athalye et al. | 2018 | 1802.00420 |
| ❌ | Reliable Evaluation of Adversarial Robustness (AutoAttack) | Croce & Hein | 2020 | 2003.01690 |

## Semi-Supervised Learning

| Status | Paper | Authors | Year | arXiv |
|--------|-------|---------|------|-------|
| ❌ | Temporal Ensembling for Semi-Supervised Learning | Laine & Aila | 2016 | 1610.02242 |
| ❌ | Mean Teachers are Better Role Models (Mean Teacher) | Tarvainen & Valpola | 2017 | 1703.01780 |
| ❌ | MixMatch: A Holistic Approach to Semi-Supervised Learning | Berthelot et al. | 2019 | 1905.02249 |
| ❌ | Unsupervised Data Augmentation for Consistency Training (UDA) | Xie et al. | 2019 | 1904.12848 |
| ❌ | FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence | Sohn et al. | 2020 | 2001.07685 |

---

**Totals:** ~313 papers | 12 ✅ converted | ~298 ❌ need download+conversion
