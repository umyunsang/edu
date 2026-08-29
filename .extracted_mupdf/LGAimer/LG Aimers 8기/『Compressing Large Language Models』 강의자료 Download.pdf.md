## --- [Page 1] ---
U Kang
1

Compressing Large Language

Models

U Kang
Head Professor, IPAI
Professor, Dept. of CSE
Seoul National University

## --- [Page 2] ---
U Kang
2

Outline

LLM Compression
PTQ & QAT
Q-Lora & QA-Lora

## --- [Page 3] ---
U Kang
3

Language Models

◼Language models show astonishing

improvements in natural language processing

❑Code generation

❑Text summarization

❑Chat

❑Translation

❑Content generation

## --- [Page 4] ---
U Kang
4

Language Models

◼However, language models are too large and

generate severe side effects

❑Examples of language models with their sizes

◼OPT-175B, LLaMA-65B, and LLaMA2-70B

❑Side effects

◼Huge storage (memory, disk) requirement

◼Computationally expensive

◼Uses lots of energy

◼Hard to deploy models on small devices (e.g., smart phones)

◼We need to compress language models to benefit

language models without side effects

## --- [Page 5] ---
U Kang
5

Language Model Compression

◼Goal: compress a language model to be fast,

memory-efficient, and energy-efficient while 
maintaining the accuracy

◼Practical constraints

❑The task is to compress an accurate pretrained model

which is trained on an extensive corpus

❑It is important to minimize the cost of compression

algorithms considering the massive scale of language 
models

## --- [Page 6] ---
U Kang
6

Compression Methods

◼Pruning

◼Distillation

◼Quantization

## --- [Page 7] ---
U Kang
7

Pruning

◼Standard deep neural network has…

❑Significant memory usage

❑Heavy power consumption
Input: Deep neural network to train
Output: Compressed deep neural network which 
preserves accuracy

## --- [Page 8] ---
U Kang
8

Pruning

◼Pruning Weights

❑Motivated by how real brain learns

❑Remove weights which 𝑤𝑒𝑖𝑔ℎ𝑡< 𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑

❑Retrain after pruning weights

❑Learn effective connections by iterative pruning

## --- [Page 9] ---
U Kang
9

Pruning

◼Compressing CNN by pruning

❑Experimented with LeNet (MNIST), AlexNet, VGGNet

(ImageNet)

❑About 10x compared to original network

## --- [Page 10] ---
U Kang
10

Knowledge Distillation

◼Based on teacher-student model

Teacher 
Model
Student
Model

cross-entropy
w/ soft target

Making soft target:

Large T makes softer distribution

The small model is trained with 
the weighted average of 
•
cross-entropy w/ teacher’s soft target
•
cross-entropy w/ correct labels

## --- [Page 11] ---
U Kang
11

◼Given a BERT model, compress it to a lightweight one

while maintaining its accuracy

◼Teacher = Original BERT Model

◼Student = Compressed BERT Model

Knowledge Distillation

## --- [Page 12] ---
U Kang
12

◼Deep Self-Attention Distillation

Knowledge Distillation

[Wang et al., MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of 
Pre-Trained Transformers, NeurIPS 2020]

## --- [Page 13] ---
U Kang
13

◼Results on SQuAD 2.0 and GLUE dev sets

❑Compaison to the state-of-the-art distillation methods

❑MiniLM gives the best performance

Knowledge Distillation

## --- [Page 14] ---
U Kang
14

◼Uniform quantization maps real values to their

corresponding 𝑏-bit integer values

❑𝑏 is the bit-width of the quantized model; generally 𝑏≤8

❑Round-to-nearest (RTN) is the basic approach of uniform

quantization

Uniform Quantization

## --- [Page 15] ---
U Kang
15

◼Quantization function of RTN

𝑄𝑟= 𝑟

Δ + 𝑧

where Δ = max 𝑟−min 𝑟

2𝑏−1
 and 𝑧= −2𝑏−1 ⋅max 𝑟+ 2𝑏−1 −1 ⋅min 𝑟

max 𝑟−min 𝑟

◼
Δ is a scaling factor

◼
𝑧 is an integer value corresponding to zero input (i.e., 𝑄(0) = 𝑧)

Round-to-nearest (RTN)

In this case,

Δ = 2.05 + 0.5

28 −1
= 2.55

255 = 0.01

𝑧= −128 ⋅2.05 −127 ⋅0.5

2.05 + 0.5
= −78

When 𝑟= 1.4,

𝑄𝑟=
1.4
0.01 −78 = 62

## --- [Page 16] ---
U Kang
16

◼When we quantize a given neural network model,

we can quantize:

❑(Weight-only quantization) only weights of the given model

❑(Activation quantization) both weights and activations of

the given model

◼Both methodologies reduce the bit-width to store

the model

◼Activation quantization has faster inference speed

than weight-only quantization since activation 
quantization does not require dequantization

Quantization Targets

## --- [Page 17] ---
U Kang
17

◼Dequantization recovers real values from the

quantized values

◼A recovered real value ෤𝑟 is calculated as follows:

෤𝑟= Δ ⋅𝑄𝑟−𝑧

❑Recap: 𝑄𝑟=

𝑟
Δ + 𝑧

◼In the case of weight-only quantization, quantized

weights should be dequantized before multiplied 
with full-precision activations

Dequantization

## --- [Page 18] ---
U Kang
18

◼Quantize both weights and activations of a model

◼Since the activations differ for

each input data, max 𝑥 and min 𝑥 are
pre-calculated using sample data set
(i.e., calibration)

Activation Quantization

## --- [Page 19] ---
U Kang
19

Outline

LLM Compression
PTQ & QAT
 PTQ Methods
 QAT Methods
Q-Lora & QA-Lora

## --- [Page 20] ---
U Kang
20

PTQ vs QAT

◼Algorithms that entail end-to-end model retraining are

extremely expensive for quantizing LLMs

◼Post-training quantization (PTQ): does not require

model retraining

❑Less accurate, but efficient

◼Quantization-aware training (QAT): requires model

retraining

❑More accurate, but inefficient

## --- [Page 21] ---
U Kang
21

SmoothQuant

◼Title: Accurate and Efficient Post-Training Quantization for Large

Language Models [ICML’ 23]

◼Activations are harder to quantize than weights since they

contain outliers that induce the quantization error

◼SmoothQuant moves the magnitude of outlier channels in

activation to the corresponding weights to reduce the difficulty of 
quantizing activation

## --- [Page 22] ---
U Kang
22

Challenge

◼How can we mitigate the difficulty of quantizing activations that

contain outliers?

◼Activations in LLMs contain large outliers in specific channels

◼Outliers significantly degrade the performance of the quantized

models by widening the quantization intervals

■This makes quantizing activations much harder than quantizing

weights

## --- [Page 23] ---
U Kang
23

Main Idea

◼Smooth the activation outliers by migrating the quantization

difficulty from activations to weights

■Scale down outlier channels in activations and scale up

corresponding weights to mitigate outliers

◼𝑾𝑿≈𝒬𝑾diag 𝒔
𝒬diag 𝒔−1𝑿, where 𝑾is weight, 𝑿is 
activation, and 𝒔 is migration parameter

■𝒔𝒋= max 𝑋𝑗

𝛼/ max 𝑊𝑗

1−𝛼, where 𝛼 is a balancing 
hyperparameter to evenly distribute the difficulties

## --- [Page 24] ---
U Kang
24

QuaRot

◼Title: Outlier-Free 4-Bit Inference in Rotated LLMs [NeurIPS ’24]

◼Rotating a vector in random direction distributes its magnitude in

all dimensions and removes an outlier

◼QuaRot quickly rotates each activation using randomized

Hadamard matrix and restores it after matrix multiplication to 
mitigate quantization error induced by outliers

## --- [Page 25] ---
U Kang
25

Challenge

◼How can we remove the outliers of activations during

quantization?

◼Activations contain outliers which increase the quantization error

◼Removing outliers without altering the result of matrix

multiplication would improve the quantization performance

## --- [Page 26] ---
U Kang
26

Main Idea

◼Rotating an activation before quantization to remove the outlier

◼Rotating a vector in the random direction probabilistically

removes outliers

◼QuaRot utilizes randomized Hadamard matrix which enables the

fast Hadamard transform for a quick random rotation

■𝑾𝑿≈𝒬𝑾𝑯−1 𝒬𝑯𝑿, where 𝑯 is a randomized Hadamard matrix

## --- [Page 27] ---
U Kang
27

Outline

LLM Compression
PTQ & QAT
 PTQ Methods
 QAT Methods
Q-Lora & QA-Lora

## --- [Page 28] ---
U Kang
28

QAT

◼Title: EfficientQAT: Efficient Quantization-Aware Training for Large

Language Models [ACL ’25]

■Current SOTA in QAT of LLMs
■Better performance compared to PTQ with affordable amount of

computational overhead (even under one A100 GPU constraint)
■Weight-only quantization
■Challenge: Efficient QAT (1. fast training, 2. accurate training)

## --- [Page 29] ---
U Kang
29

QAT

◼EfficientQAT [ACL ’25]

■How? Two-stage approach
■0. RTN (Round-To-Nearest)
■1. Block-AP: Sequential block-wise training of all parameters (𝑠, 𝑧, 𝑾)
■2. E2E-QP: End-to-end training of quantization parameters (𝑠)
■Existing QAT (E2E-AP): End-to-end training of all parameters

** scaling factor 𝑠, zero-point 𝑧, and weight matrix 𝑾

## --- [Page 30] ---
U Kang
30

QAT

◼EfficientQAT [ACL ’25]

■Why does it work? Block-AP provides both efficient and effective

initialization for the following end-to-end training (E2E-QP)

■➔ Total of 41 hours to quantize a ~70B model with QAT

## --- [Page 31] ---
U Kang
31

Outline

LLM Compression
PTQ & QAT
Q-Lora & QA-Lora

## --- [Page 32] ---
U Kang
32

PEFT

◼Existing high-cost compression algorithms for small

language models show superior performance 
through repeated fine-tuning

❑Fine-tuning updates all parameters of the LLMs using a

task-specific objective function and data

◼However, LLMs are too large to fine-tune the entire

model

❑GPT-3 175B requires 1.2TB VRAM to update the entire

model

◼
An 80GB A100 GPU costs more than 15,000$

◼How can we effectively fine-tune massive-scale

LLMs?

## --- [Page 33] ---
U Kang
33

PEFT

◼PEFT (Parameter Efficient Fine-Tuning) is an efficient

algorithm to fine-tune LLMs

❑When fine-tuning a pre-trained LLM using PEFT, only a

small number of (additional) model parameters are 
modified.

❑PEFT allows previous accurate high-cost algorithms to be

applied for LLMs

## --- [Page 34] ---
U Kang
34

LoRA

◼LoRA (Low-Rank Adaptation) leverages low-rank

approximation to reduce the cost of fine-tuning

◼Overview of LoRA

❑(1) Parameter-efficient fine-tuning

◼
LoRA divides the fine-tuned parameters into pre-trained 
parameters and update target parameters

❑(2) Low-rank approximation

◼
LoRA applies low-rank approximation to update target parameters

[Hu et al. ICLR 2022]

## --- [Page 35] ---
U Kang
35

LoRA

◼LoRA reformulates the fine-tuning process

❑Standard full fine-tuning

ℒ𝑋, 𝑦; Θ0 
ℒ𝑋, 𝑦; Θ

❑Fine-tuning in a parameter-efficient approach

ℒ𝑋, 𝑦; Θ0 
ℒ𝑋, 𝑦; Θ0 + ΔΘ

◼
ℒ is the task-specific loss function

◼
𝑋 and 𝑦 represent the data and its corresponding label, 
respectively

◼
Θ refers to the fine-tuned parameters

◼
Θ0 and ΔΘ are the parameters from the pre-trained model and the 
update target parameters, respectively

[Hu et al. ICLR 2022]

## --- [Page 36] ---
U Kang
36

LoRA

◼LoRA applies low-rank approximation to the update

target weight Δ𝑾∈ℝ𝑑×𝑑, not to the pretrained 
weight 𝑾∈ℝ𝑑×𝑑

𝒉= 𝑾𝒙+ Δ𝑾𝒙= 𝑾𝒙+ 𝑩𝑨𝒙

❑𝒙∈ℝ𝑑 is an input and 𝒉∈ℝ𝑑 is an output

❑𝑨∈ℝ𝑟×𝑑 and 𝑩∈ℝ𝑑×𝑟 are low-rank matrices to approximate Δ𝑾

▪
𝑟 is a rank of low-rank approximation

• LoRA adapts only the MHA sublayer 𝑊∈𝑊𝑞, 𝑊𝑘, 𝑊𝑣, 𝑊𝑜

[Hu et al. ICLR 2022]

## --- [Page 37] ---
U Kang
37

LoRA

◼The most significant benefit of LoRA comes from the

reduction in memory footprint

❑On GPT-3 175B, LoRA reduces the VRAM consumption

during training from 1.2TB to 350GB

◼
We do not need to store the optimizer states for the frozen 
parameters

❑The checkpoint size is reduced from 350GB to 35MB

(roughly 10,000×) with rank 𝑟= 4, and only the query and 
value projection matrices are adopted

[Hu et al. ICLR 2022]

## --- [Page 38] ---
U Kang
38

LoRA

◼LoRA shows comparable performance compared to

fine-tuning (FT) across the entire model, despite 
tuning < 0.022% of the parameters

❑The ideal parameter budget varies across datasets

◼
On the MNLI-m dataset, LoRA with 4.7M parameters outperforms 
LoRA with 37.7M parameters

* Accuracy on WikiSQL and MNLI-m datasets after fine-tuning

[Hu et al. ICLR 2022]

## --- [Page 39] ---
U Kang
39

LoRA

◼Adapting LoRA to all weight matrices in the MHA

sublayer outperforms other cases

❑We allocate a parameter budget of 18M for all 96 layers of

GPT-3 175B.

* Accuracy on WikiSQL and MultiNLI datasets after fine-tuning GPT-3 175B with 
18M parameters

• 𝑊𝑞, 𝑊𝑘, 𝑊𝑣, and 𝑊𝑜 are query, key, value, and out projection matrices, respectively

[Hu et al. ICLR 2022]

## --- [Page 40] ---
U Kang
40

LoRA to QLoRA

◼LoRA: How to fine-tune a pretrained model quickly?

❑Input: a pretrained model ℳ (full-precision)

❑Output: a finetuned model ℳ (full-precision)

◼QLoRA: How to fine-tune a pretrained model with

less memory overhead?

❑Input: a pretrained model ℳ (full-precision) and

quantization bit 𝐵

❑Output: a 𝐵-bit quantized model ෡
ℳand adapters (full-
precision)

https://github.com/artidoro/qlora

## --- [Page 41] ---
U Kang
41

FP16

FP16

NF4

FP16

INT4

FP16

FP16

NF4

INT4

FP16

FP16

FP16

FP16

FP16

Training
Inference

𝑠

𝑧

𝑠
𝑠

𝑠

𝑧

QLoRA

LoRA

QALoRA

QLoRA

## --- [Page 42] ---
U Kang
42

QLoRA (Idea1: 4-bit NormalFloat)

◼How to correctly express weights with limited bit

widths?

❑Determine quantization levels based on weight distribution

(Normal distribution)

❑Use codebook

## --- [Page 43] ---
U Kang
43

QLoRA (Idea2: Double Quantization)

◼How to decrease the memory for the scale

parameters?

❑Each block (64 parameters) requires to store a FP32 scale

parameter

❑Double quantization: quantize the scales one more time =>

◼
256 scales are quantized into FP8 with FP32 scale

❑Overhead:

32
64 = 0.5 𝑏𝑖𝑡𝑠→
8
64 +
32
(64∗256) = 0.127𝑏𝑖𝑡𝑠

FP32

FP32

FP32

FP32

FP8

FP8

FP8

FP8

FP32

Weight Scale
Weight Scale

Scale’s Scale


|  |  |
| --- | --- |
| N | F4 |

## --- [Page 44] ---
U Kang
44

QLoRA to QALoRA

◼QLoRA: How to fine-tune a pretrained model with

less memory overhead?

❑Input: a pretrained model ℳ (full-precision) and

quantization bit 𝐵

❑Output: a 𝐵-bit quantized model ෡
ℳand adapters (full-
precision)

◼QALoRA: How to fine-tune a pretrained model to

achieve fast inference with less memory overhead?

❑Input: a pretrained model ℳ (full-precision) and

quantization bit 𝐵

❑Output: a 𝐵-bit quantized model ෡
ℳ

## --- [Page 45] ---
U Kang
45

FP16

FP16

NF4

FP16

INT4

FP16

FP16

NF4

INT4

FP16

FP16

FP16

FP16

FP16

Training
Inference

𝑠

𝑧

𝑠
𝑠

𝑠

𝑧

QLoRA

LoRA

QALoRA

QALoRA

## --- [Page 46] ---
U Kang
46

QALoRA

◼How to remove adapters after QLoRA-training?

❑Merge LoRA weight to zero-points of quantized

weights

❑One LoRA weight per each quantization group

Group-wise Quantization

𝑑

broadcasting

https://github.com/yuhuixu1993/qa-lora


|  |  | 𝑟 |  |
| --- | --- | --- | --- |
| 𝑑 𝑔 |  |  |  |
|  |  |  |  |

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| × |  | 𝑟 |  |  |  |  |
|  |  |  |  |  |  |  |

|  |  |  |  |  |  |  |  | 𝑟 |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  | + 𝑑 |  |  |  |  | ×𝑟 |  |  |  |  |
| 𝑑 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## --- [Page 47] ---
U Kang
47

Conclusion

◼Language model compression: compress a

language model to be fast, memory-efficient, and 
energy-efficient while maintaining the accuracy

◼Quantization method is effective for LLM

compression

◼PTQ and QAT methods have tradeoff in accuracy

and efficiency

◼Parameter-efficient file tuning methods enable

effective training

## --- [Page 48] ---
U Kang
48

Thank You!