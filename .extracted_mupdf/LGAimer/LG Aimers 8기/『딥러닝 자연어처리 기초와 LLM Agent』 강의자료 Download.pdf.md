## --- [Page 1] ---
LG AI Research Special Lecture

Lecture 01 – AI의 첫걸음, 머신러닝과 딥러닝의 기초

중앙대학교 AI학과

이환희

딥러닝 자연어처리 기초와 LLM 에이전트


|  |  |  |  |
| --- | --- | --- | --- |
|  |  |  | 1 |

## --- [Page 2] ---
2

Outline

• The AI Landscape and Machine Learning

• Types of Machine Learning

• Deep Learning Fundamentals

AI의 첫걸음, 머신러닝과 딥러닝의 기초

## --- [Page 3] ---
3

Machine Learning

• Machine Learning: statistical algorithms that can effectively generalize and thus perform tasks without 
explicit instructions.

## --- [Page 4] ---
4

Learn from Experience

Slides adapted from STAT 451

Task T: classifying handwritten digits from images

Performance measure P : percentage of digits classified correctly

Training experience E: dataset of digits given classifications, e.g., MNIST

## --- [Page 5] ---
5

Machine Learning?

• Sub-field of AI that relates to making computers learn from experience.

• Requires:

• Data

• Used in learning (training) how to accomplish some task

• Features

• Algorithm

•
Improves automatically through experience (training)

## --- [Page 6] ---
6

Types of Machine Learning

• Algorithm requires a feedback during training (learning) phase.

• Machine Learning types, by the nature of that feedback:

•
Supervised

•
Unsupervised

•
Semi-supervised

•
Reinforcement

## --- [Page 7] ---
7

Three Types of Machine Learning

## --- [Page 8] ---
8

Supervised Machine Learning

• Training data: labeled (ground truth - expected algorithm result; expensive!)

• Training: Algorithm uses labels to evaluate its accuracy on training data.

• Risk of overfitting.

## --- [Page 9] ---
9

Supervised Learning

## --- [Page 10] ---
10

Supervised Machine Learning

• Two types of problems it tries to solve:

• Regression

• Predict numerical (continuous) value

• Linear, Nonlinear Regression

• Classification:

• Predict categorical (discrete) value

• Naive Bayes Classifier, Support Vector Machines, Logistic Regression, ...

•
Decision Tree, Random Forest, k-NN, Neural Networks, etc...can solve both problems

## --- [Page 11] ---
11

Supervised Learning Workflow

## --- [Page 12] ---
12

Semi-supervised Machine Learning

• Hybrid learning - between supervised and unsupervised

• Solves the problem of having not enough labeled data to train a supervised learning algorithm

• Training data: small labeled and large unlabeled data set

• Training:

• Train model with labeled data

• Trained model used to predict

•
labels for unlabeled data =>

•
pseudo-labeled data

• Retrain model with both pseudo- and labeled data sets

## --- [Page 13] ---
13

Unsupervised Machine Learning

• Training data: Unlabeled data

• Training:

• Extract features and patterns from data itself

• Clustering: these features used to label and

classify the data into clusters

• Methods: k-Means clustering, ...

## --- [Page 14] ---
14

Unsupervised Learning

## --- [Page 15] ---
15

Reinforcement Machine Learning

• Training data: none

• Training:

• Machine trained to make specific decisions

• Machine interacting with its environment

• Trial and error

• Reward system: providing feedback when an artificial intelligence agent performs the best action in a particular

situation

• Sequence of successful outcomes is reinforced to develop the best solution for a given problem.

## --- [Page 16] ---
16

What is Deep Learning?

• Deep Neural Networks - more than one hidden layer (vs Shallow Network)

• With each new hidden layer system becomes

• more intelligent;

• increases capabilities to learn: new features and

more complex features

• Each feature reflects one detail from the input

## --- [Page 17] ---
17

Artificial Neural Network

• Collection of simple, trainable, interconnected mathematical 
units (neurons) that collectively learn complex function

• System of interconnected layers:

•
input: numeric representation of the data

•
hidden: nonlinear function (activation function) of the sum of

weighted inputs from the previous layer plus bias

•
output: prediction - set of values (continuous ⇒model solves

regression problem; discrete ⇒classification

y = F(𝑤1 x 𝑥1 + 𝑤2 x 𝑥2 + ... 𝑤𝑁x 𝑥𝑁+ b);

F(x) = max(0, x); (ReLU)

𝑥𝑖- input, 𝑤𝑖- weight

## --- [Page 18] ---
18

What is Learning in Neural Network?

• Learning (training):

• Auto-adjusting model parameters with the each

new input sample so the output (prediction) gets

closer to expected values (ground truth)

• Forward-propagating activations for labeled

input and back-propagating

• Errors to adjust parameters in each node in

order to minimize loss function

## --- [Page 19] ---
19

Neural Network Input: Digital Image

• Digital image - 2D set of pixels.

• Each pixel has numeric value(s) 
associated to it:

• Monochrome: 0, 1

• Grayscale: 0 (black) - 255 (white)

• Colour: 0 - 25 (in each channel: R, G, B)

• Computer sees an array of numbers.

## --- [Page 20] ---
20

Fully-connected Input Layer

•
Feeding input to a fully-connected input layer:

•
2D intensity (or 3D colour) matrix is collapsed into 1D vector

## --- [Page 21] ---
21

XOR Problem

•
Minsky-Papert proved perceptron can’t compute XOR logical operation

## --- [Page 22] ---
22

XOR Problem

•
Perceptron can compute the logical AND and OR functions easily

•
But it’s not possible to build a perceptron to compute logical XOR!

## --- [Page 23] ---
23

XOR Problem

•
Perceptron is a linear classifier but XOR is not linearly separable

## --- [Page 24] ---
24

Why do we need non-linear activation functions?

•
Network of simple linear (perceptron) units cannot solve XOR problem

•
A network formed by many layers of purely linear units can always be reduced to a single layer of linear units

•
We’ve already shown that a single unit cannot solve the XOR problem

## --- [Page 25] ---
25

XOR Problem with Non-linear Functions

•
XOR function can be computed using two layers of ReLU-based units

•
XOR problem demonstrates need for multi-layer networks

## --- [Page 26] ---
26

Nonlinearity Layer: Rectified Linear Unit (ReLu)

• Activation function: nonlinear function converts feature map into the activation map

•
ReLU makes CNNs perform best.

•
CNN data input is usually normalized to range [-1, 1] =>
improves learning speed and accuracy

## --- [Page 27] ---
27

Fully-connected layers

• They learn non-linear combinations of the high-level features outputted by the convolutional layers.

• Features are assign weights which describe their contribution in recognizing particular object.

## --- [Page 28] ---
28

Softmax Layer

•
Serving as the final layer after the Fully Connected layer, the Softmax function converts the network's raw scores

(logits) into a meaningful probability distribution (summing to 1) for each class.

•
This probability output is essential for calculating the "error" (Loss), which is then used by the backpropagation

algorithm to update the network's weights during training.

Last
Hidden

Layer

Predictions
Softmax
activation

function

## --- [Page 29] ---
29

How Do Neural Networks Learn? Backpropagation

•
To minimize the "Error" (or "Loss") by adjusting the network's internal parameters (weights). The error 
is the difference between the network's Prediction and the Actual Target (the correct answer).

1. Forward Pass (Make a Guess)
•
Input data is fed forward through the network (Input Layer → 
Hidden Layers → Output Layer).
•
The network makes a Prediction.
•
The Error is calculated by comparing the prediction to the correct 
answer.

2. Backward Pass (Learn from the Mistake)
•
The Error is propagated backward from the Output Layer to the 
Input Layer.
•
Backpropagation algorithm calculates how much each Weight in 
the network contributed to the total error.
•
These weights are then Adjusted (updated) slightly in a direction 
that will reduce the error.
•
"Forward Pass → Backward Pass" cycle is repeated many times, 
allowing the network to get progressively better at its task.

https://botpenguin.com/glossary/backpropagation

## --- [Page 30] ---
30

Summary

• Aritifical Intelligence is the broad concept of machines mimicking human behavior, Machine Learning is a subset of AI

that learns from data, and Deep Learning is a subset of ML using multi-layer neural networks.

• Supervised Learning learns from labeled data to solve problems like Classification (predicting a category) and Regression

(predicting a continuous value).

• Unsupervised Learning discovers hidden patterns and structures in unlabeled data, with a primary task being Clustering

• Reinforcement Learning involves an "agent" learning by maximizing a reward signal, while Semi-supervised Learning

uses a mix of labeled and unlabeled data.

• The classic XOR problem proved the limitations of single-layer linear models (Perceptrons), demonstrating the need for

multi-layer networks equipped with non-linear activation functions (e.g., ReLU) to solve complex problems.

• Backpropagation is a core training algorithm for neural networks; it's a cycle of making a prediction (Forward Pass),

calculating the Error (Loss), and then adjusting the network's weights to minimize that error (Backward Pass).

• Softmax Layer is used as the final layer in classification tasks, it converts the network's raw scores (logits) into a

meaningful probability distribution (summing to 1) across all classes.

## --- [Page 31] ---
LG AI Research Special Lecture

Lecture 02 – 자연어처리의 기초와 RNN

중앙대학교 AI학과

이환희

딥러닝 자연어처리 기초와 LLM 에이전트


|  |  |
| --- | --- |
|  | 31 |

## --- [Page 32] ---
32

Outline

자연어처리의 기초와 RNN
•
What is Natural Language Processing?

•
Tokenization

•
Word Embedding

•
Language Modeling

•
Recurrent Neural Networks

## --- [Page 33] ---
33

Natural Language Processing

NLU
NLG

Natural Language Understanding

Natural Language Processing
NLP

Natural Language Generation

•
Giving computers the ability to understand text in much the same way human beings can.

https://www.ibm.com/topics/natural-language-processing

## --- [Page 34] ---
34

Natural Language Processing Tasks

Machine Translation

Summarization
Dialog System

Story Generation

## --- [Page 35] ---
35

Building NLP Systems

Preprocessing
Embedding
Modeling

Breaking up the 
input text into 
individual words or 
tokens. (tokenization)

Train the NLP system 
on a large corpus of 
text data, allowing it to 
make predictions and 
classify new text.

Represent words in 
a numerical vector 
space.

Input Text: “I loved the 
movie”

Output Tokens: [“I“, 
“loved", “the", “movie“]

N x D vectors

N: num of words

D: dimension

“Positive”

## --- [Page 36] ---
36

Tokenization: Example

## --- [Page 37] ---
37

Tokenization: Padding

## --- [Page 38] ---
38

Tokenization: Beyond Word Level – Vocab Size?

• What happens if the vocab size is too big? Or too small?

Solution 2: Byte Pair Encoding
(3rd week’s class)

Small: Out-of-Vocabulary

“not present in a system's dictionary or 
database of known terms”

Big: Too-much computation!, 
sparsity

Solution 1: Character Level

## --- [Page 39] ---
39

Word Embedding

• A representation that maps words to real-valued vectors

## --- [Page 40] ---
40

Word Embedding

•
Bag-of-Words(BoW)

▪
A text is represented as the bag of its words

▪
ex) One-hot encoding

dog

1

0

0

cat
1

0

pizza
0

0
0

1

## --- [Page 41] ---
41

Word Embedding

•
Word2Vec

▪
One of the most popular techniques (Tomas Mikolov, 2013)

▪
Constructs word embeddings where words with similar context are embedded close to each other

▪
CBOW and Skip-gram models

I want              food for lunch 
?

Korean

Italian
desk

book

## --- [Page 42] ---
42

Word Embedding

•
How to generate word embedding in Word2Vec

▪
Center word and its context words

Wanting  less  feels  like  getting  more

Window for context

(window size = 3)

Center word

## --- [Page 43] ---
43

Word Embedding

𝐸= −log 𝑝(𝑤𝑡−𝑐, … 𝑤𝑡+𝑐|𝑤𝑡)
𝐸= −log 𝑝(𝑤𝑡|𝑤𝑡−𝑐, … 𝑤𝑡+𝑐)

◼CBOW(Continuous Bag-of-Words)
◼Skip-gram

## --- [Page 44] ---
44

Word Embedding

• Skip-gram model

▪𝑦𝑗= 𝑝𝑤𝑗𝑤𝑖is the probability that 𝑤𝑗is the context word, given the input 𝑤𝑖

0

1

0

0

0

0

wanting

less

feels

like

getting

More

wanting

less

feels

like

getting

More


|  |  |  |
| --- | --- | --- |
|  | 𝑢 𝑒 𝑗 𝑦 = 𝑗 𝑉 𝑢 σ 𝑒 𝑗′ 𝑗′=1 |  |
|  |  |  |

|  | 𝑇 ℎ = 𝑊 𝑥 |  |
| --- | --- | --- |


|  | 𝑇 𝑢 = 𝑊′ ℎ |
| --- | --- |


## --- [Page 45] ---
45

Word Embedding

• Skip-gram model

▪Rows in hidden layer weight matrix become word vectors

(word vector look-up table)

0

1

0

0

0

0

wanting

less

feels

like

getting

More

𝑥
ℎ

V

d = # Hidden units = features
[V x 1]

[d x 1]

𝑊

## --- [Page 46] ---
46

Language Model

• A model that has learned the probability distribution of sentences or words.

• By learning from a large amount of text data, the model understands the patterns and characteristics of

sentences, and based on this, it can generate new sentences or predict the next word.

## --- [Page 47] ---
47

Language Model

• Language Modeling: Task of understanding the probability distribution over a sequence of words

I
am
a
good

am
a
good
student

## --- [Page 48] ---
48

Language Model

•
A language model computes a probability for a sequence of words:

P(w1, … , wT)

Word ordering:
P(the cat is small)  >  P(small the is cat)

Word choice:
P(walking home after school)  >  P(walking house after school)

To compute  P(w1, … , wT)

= P w1 P w2 w1 P w3 w1, w2 … P(wn|w1, w2, … wn−1)

P w1

SOS

P w2|𝑤1

𝑤1

P w3|𝑤1, 𝑤2

𝑤2

P w4|𝑤1, 𝑤2, 𝑤3

𝑤3

……

## --- [Page 49] ---
49

Building Neural Language Model: Fixed Language Model

Slides adapted from CS224n

•
Input: sequence of words  -
Output: prob. dist. of the next word

•
A simple fixed-length language model predicts the probability of the next word by considering only a constant,

predefined number of preceding words (a "fixed window"), rather than the entire sequence history.

“We need a neural architecture 
that can process any length input”

## --- [Page 50] ---
50

Recurrent Neural Network

•
RNNs are very natural way to model sequential data:

•
RNN consider input sequence over discrete time.

•
RNN have the ability to remember information in their hidden states for a long time.

•
They are equivalent to very deep nets with a hidden layer per one time step.

•
Except that they use the same weights at every time step and they get input at every time step.

Apply the same weights repeatedly!

## --- [Page 51] ---
51

Simple RNN Language Model

Slides adapted from CS224n

Input sequence 
could be longer

## --- [Page 52] ---
52

Pros and Cons of RNN

• Advantages of RNN:

• Can process any length input laptops

• Computation for step t can use information from many steps back

• Model size doesn’t increase for longer input context

• Same weights applied on every timestep, so there is symmetry in how inputs are processed.

• Disadvantages of RNN

• Recurrent computation is slow

• In practice, difficult to access information from many steps back

Slides adapted from CS224n

## --- [Page 53] ---
53

Training an RNN Language Model

• Get a big corpus of text which is a sequence of words

• Feed into RNN-LM and compute output distribution          for every step t.

• i.e., predict probability dist of every word, given words so far

• Loss function on step t is cross-entropy between predicted probability distribution , and the true next

word (one-hot for ):

• Average this to get overall loss for entire training set

Slides adapted from CS224n

## --- [Page 54] ---
54

Training an RNN Language Model

Slides adapted from CS224n

## --- [Page 55] ---
55

Training an RNN Language Model

Slides adapted from CS224n

“Teacher Forcing”

## --- [Page 56] ---
56

Training an RNN Language Model

• Computing loss and gradients across entire corpus                          at once is too expensive!

• In practice, consider                            as a sentence (or a document)

• Recall: Stochastic Gradient Descent allows us to compute loss and gradients for small chunk of data, 
and update.

• Compute loss 𝐽(𝜃) for a sentence (actually, a batch of sentences), compute gradients and update 
weights.

• Repeat on a new batch of sentences.

Slides adapted from CS224n

## --- [Page 57] ---
57

Backpropagation of RNNs

•
Derivative of 𝐽(𝑡)(𝜃) with respect to the repeated weight matrix:

•
The gradient w.r.t. a repeated weight is the sum of the gradient w.r.t. each time it appears.

Slides adapted from CS224n

## --- [Page 58] ---
58

Training the parameters of RNNs: Backpropagation for 
RNNs

Slides adapted from CS224n

## --- [Page 59] ---
59

Generating with an RNN Language Model (“Generating 
roll outs”)

•
We can use a RNN language model to generate text by repeated sampling.

•
Sampled output becomes next step’s input.

Slides adapted from CS224n

## --- [Page 60] ---
60

Problem of RNN: Vanishing Gradients

• Vanishing Gradients problem for basic RNNs

▪
Influence of the inputs at time t decreases and vanishes over time

## --- [Page 61] ---
61

LSTM-RNNs

• LSTM can preserve gradient information

▪
Hidden layer units formed with Long Short-Term Memory (LSTM) cells can store

and access information over longer periods of time.

## --- [Page 62] ---
62

LSTM-RNNs

• LSTM block architecture

▪
3 gates

▪
Input gate adjust the influence

from input to cell

▪
Forget gate adjust the influence

from cell to cell over time

▪
Output gate adjust the influence

from cell to output

## --- [Page 63] ---
63

RNN for Encoding Sentence

Slides adapted from CS224n

• Usage on sentiment classification

## --- [Page 64] ---
64

Summary

•
NLP involves preprocessing text, representing words as numerical vectors, and training models to understand and 
generate language.

•
Tokenization breaks raw text into meaningful units (tokens) like words or sub-words, which are then mapped to 
indices in a vocabulary.

•
Word Embeddings (like Word2Vec) learn dense, low-dimensional vectors for these tokens, capturing semantic 
meaning by placing words with similar contexts close together in the vector space.

•
Language Modeling is the task of learning the probability of a sequence of words, which is fundamental for predicting 
the next word or generating new text.

•
Recurrent Neural Networks (RNNs) are designed to process sequences of any length by using an internal hidden state 
that acts as a memory, carrying information from one timestep to the next.

•
RNNs are trained using Backpropagation Through Time, which unrolls the network and sums gradients at each 
timestep to update the shared weights.

•
Simple RNNs struggle with long-term dependencies due to the vanishing gradient problem, which is why more 
complex architectures like LSTMs use internal gates to better control the flow of information and preserve memory.

## --- [Page 65] ---
LG AI Research Special Lecture

Lecture 03 – LLM의 핵심, 트랜스포머와 어텐션 메커니즘

중앙대학교 AI학과

이환희

딥러닝 자연어처리 기초와 LLM 에이전트


|  |  |
| --- | --- |
|  | 65 |

## --- [Page 66] ---
66

Outline

•
Seq2seq Model

•
Attention Mechanism

•
Self-Attention and Transformer

•
Language Generation with Transformer

LLM의 핵심, 트랜스포머와 어텐션 메커니즘

## --- [Page 67] ---
67

The Evolutionary History of Language Modeling

Seq2Seq
Seq2Seq
with Attention
Transformer

Encoder-Decoder

구조의 모델

Decoding 과정에서의

Adaptive encoding

Source sequence → Target sequence

- Machine translation / Dialog generation

- Parsing sentences into grammar trees

Self-Attention
Multi-head Attention

## --- [Page 68] ---
68

Seq2Seq Model

https://web.stanford.edu/class/cs224n/

## --- [Page 69] ---
69

Seq2Seq Encoder-Decoder using RNN

•
We need to understand seq2seq encoder-decoder model to know the motivation of ‘attention mechanism’.

•
Encoder: from word sequence to sentence representation (a real-valued vector).

•
Decoder: from representation to word sequence distribution

← Word embedding

## --- [Page 70] ---
70

The Evolutionary History of Language Modeling

Seq2Seq
Seq2Seq
with Attention
Transformer

Encoder-Decoder

구조의 모델

Decoding 과정에서의

Adaptive encoding

Source sequence → Target sequence

- Machine translation / Dialog generation

- Parsing sentences into grammar trees

Self-Attention
Multi-head Attention

## --- [Page 71] ---
71

Attention Model – Motivation

• Challenge in vanilla seq2seq for long sentences

• Decoder generates a translation solely based on the last hidden state.

• Information about the first word needs to be encoded in the last hidden state.

x1
xT
…

←     50 words     →

h1
hT
…
…

y1

## --- [Page 72] ---
72

Intuition of Attention Mechanism

•
Attention mechanism in decoder

•
The decoder decides which different parts of the source sentence to  pay “attention” at each step of 
the output generation.

## --- [Page 73] ---
73

Attention Mechanism with Seq2Seq

•
RNN hidden state of the decoder at i:

•
The context vector 𝑐𝑖is computed as a weighted sum of annotations ℎ1, … , ℎ𝑇:

•
How to get attention weight 𝛼𝑖𝑗:

Alignment score function

Context vector ct

𝑠𝑖= 𝑓𝑠𝑖−1, 𝑦𝑖−1, 𝑐𝑖

## --- [Page 74] ---
74

Attention Mechanism – Scoring

• Alignment score function

where 𝑠𝑖−1 is the RNN hidden state just before emitting 𝑖 th word,

and ℎ𝑗is the 𝑗 th RNN hidden state of the input sentence.

• It scores how well the inputs around position 𝑗 and the output at position 𝑖 match.

𝑒𝑖𝑗= score 𝑠𝑖−1, ℎ𝑗

## --- [Page 75] ---
75

Attention Mechanism – Normalization

• Let 𝛼𝑖𝑗 be the probability that the target word 𝑦𝑖 is aligned to (or translated from) a source word 𝑥𝑗.

• 𝛼𝑖𝑗 is computed by normalizing the probabilities with a softmax:

## --- [Page 76] ---
76

Limitation of RNN

• Difficulty in capturing long-term dependencies:

▪Suffer from the vanishing gradient problem

• Computationally expensive

▪Difficult to train

▪Very long gradient paths

• Inability to process parallel sequences

▪Can only process one input at a time.

## --- [Page 77] ---
77

The Evolutionary History of Language Modeling

Seq2Seq
Seq2Seq
with Attention
Transformer

Encoder-Decoder

구조의 모델

Decoding 과정에서의

Adaptive encoding

Source sequence → Target sequence

- Machine translation / Dialog generation

- Parsing sentences into grammar trees

Self-Attention
Multi-head Attention

## --- [Page 78] ---
78

Self Attention

•
Self-Attention: A technique that calculates the influence of each word on each other within an input sentence 
and weights it accordingly, and uses it to obtain a new presentation for each word

•
Each element in the sentence attends to other elements from the same sentence -> context-sensitive encodings

•
Self-attention enhances the automatic understanding of text

## --- [Page 79] ---
79

Self Attention & Transformer

•
Self-Attention: A technique that calculates the influence of each word on each other within an input 
sentence and weights it accordingly, and uses it to obtain a new presentation for each word

•
Transformer: Multi-layered artificial neural network model using Self-Attention technique

https://jalammar.github.io/illustrated-transformer/

Representation for "it" 
is obtained by intensive 
reference to "animal."

## --- [Page 80] ---
80

RNN vs Transformer

•
Transformer

•
Self-attention mechanism : Consists
of fixed-size matrix multiplications

Hard to parallelize efficiently (seq

uential nature)

Easy to parallelize → Fast

No explicit modeling of long-
range dependencies

Complete connections
between consecutive layers

Maximum path length : 𝐎(𝐧)
Maximum path length : 𝐎(𝟏)

distance between any pair of input and output positions in networks

𝒏: sequence length

•
RNN

•
Recurrent module : Suitable for pro
cessing variable length representati
ons

## --- [Page 81] ---
81

Transformer: Model Overview

Soy
estudiante

I
am
a
student

INPUT

OUTPUT

Positional 
Embedding

Linear

Softmax

Positional 
Embedding

Encoder Layer 2

Encoder Layer 6

Self-Attention

Feed Forward

Network

Decoder Layer 6

Decoder Layer 1

Encoder Layer 1

Decoder Layer 2

Self-Attention

Feed Forward

Network

Enc-Dec Attention

## --- [Page 82] ---
82

•
Positional Embedding

▪Information about the relative or absolute position of the tokens in the sequence

(to utilize the order of the sequence)

Encoder Layer 2

Encoder Layer 1
(𝑝𝑜𝑠= position, 𝑖= dimension)

Transformer: Positional Embedding

soy
estudiante

𝐱𝟏
𝐱𝟐

Input token

Token
Embedding

Positional 
Embedding
𝐩𝟏
𝐩𝟐

Input
Embedding

+
+

=

=

𝐞𝟏
𝐞𝟐

Encoder Layer 2

Encoder Layer 1

## --- [Page 83] ---
83

Transformer: Encoder

•
Encoder

A stack of N=6 identical layers

Each layer has two sub-layers

(1) Multi-Head Self-attention
(2) Position-wise Feed Forward Network

The output of each sub-layer

= LayerNorm ( 𝑥+ Sublayer(𝑥))

residual connection

All of the queries, keys, and values come from the out

put of the previous layer (Q=K=V)

## --- [Page 84] ---
84

Transformer: Self-Attention

•
Self-attention (Scaled Dot-product Attention)

▪Computing a representation of a sequence considering the relationship between different

positions

1

5

4

3

2

1

0.3

0.1

0.2

Softmax

Dot-product

Query(𝐐𝐢)
Key(𝐊)
Value(𝐕)

5

2

1

The animal didn’t cross the …

[1]
[2]
[3]
[4]
[5]

*

*

*

1’


| 0.1 | * | 3 |
| --- | --- | --- |


| 0.3 | * | 4 |
| --- | --- | --- |


## --- [Page 85] ---
85

Transformer: Self-Attention

•
Multi-Head Attention

▪Apply the “Scaled Dot-Product Attention” several times with different linear projections

Linear ℎ𝑞

Linear 2𝑞

Linear 1𝑞

Linear ℎ𝑘

Linear 2𝑘

Linear 1𝑘

Linear ℎ𝑣

Linear 2𝑣

Linear 1𝑣

Query(𝐐)
Key(𝐊)
Value(𝐕)

## --- [Page 86] ---
86

Transformer: Self-Attention

The

animal

didn’

t cro

ss th

e stre

et

becaus

e it

wa

s t

oo

tired

.

The

anima

l didn’

t cros

s the

street

becaus

was

too

tired

.

The

anima

l didn’

t

cross

the s

treet

becaus

e it

was

too

tired

.

didn’

t cros

s the

stree

t

becaus

e it

wa

s t

oo

tired

.

•
Self-Attention Visualization

Single-Head Self-Attention

Multi-Head(8) Self-Attention


| The | animal |
| --- | --- |


|  |  |
| --- | --- |
|  |  |
|  |  |
|  |  |
|  |  |
|  |  |
| it |  |
| e |  |

## --- [Page 87] ---
87

Transformer: Encoder Architecture

•
Encoder

Multi-Head
Self-Attention

Feed Forward

Network

Feed Forward

Network

soy
estudiante

𝐱𝟏
𝐱𝟐

𝐳𝟏
𝐳𝟐

𝐡𝟏
𝐡𝟐

Encoder

Layer 1

Encoder

Layer 2

◼Point-wise Feed Forward Network

Applied to each position separately and id

entically (different parameters from layer t
o layer)

Two linear transformations

with ReLU activation in between

## --- [Page 88] ---
88

Transformer: Residual Connection

•
Residual Connection

Multi-Head Self-Attention

Feed Forward

Network

Feed Forward

Network

soy

𝐱𝟏
𝐱𝟐

Positional 
Embedding

Add & Norm

𝐳𝟏
𝐳𝟐

𝟏
𝐳′

𝟐
𝐳′

Add & Norm

Encoder

Layer1

𝐞𝟏
𝐞𝟐

Residual 
Connection

* Residual Connection
- Skip connection
- Short-cuts to jump ov

er some layers

* Layer Normalization
- Normalizes the inputs

across the features


|  |  |  |
| --- | --- | --- |
|  |  |  |

|  |  |  |
| --- | --- | --- |
|  |  |  |

## --- [Page 89] ---
89

Transformer: Decoder

•
Decoder (almost the same as encoder)

A stack of N=6 identical layers

Each layer has three sub-layers

(1) Multi-Head Self-attention
(2) Multi-Head Attention over the output

of the encoder stack (enc-dec attention)
(3) Position-wise Feed Forward Network

Encoder’s

output

The queries Q come from the previous decod
er layer, and the key K and value V come from
the output of the encoder

## --- [Page 90] ---
90

Overall Architecture

soy
estudiante

𝐱𝟏
𝐱𝟐

Positional 
Embedding

Multi-Head Self-Attention

Feed Forward

Network

Feed Forward

Network

Add & Norm

Add & Norm

Encoder

Layer 1

Feed Forward

Network

Feed Forward

Network

Add & Norm

Multi-Head Self-Attention

Add & Norm

Encoder

Layer 6

Multi-Head Self-Attention

Feed Forward

Network

Feed Forward

Network

Add & Norm

Add & Norm

Decoder

Layer 1

Encoder-Decoder Attention

Add & Norm

Decoder Layer 6

Linear

Softmax

Only Key(𝐊) & Value(𝑽)

## --- [Page 91] ---
91

Natural Language Generation with Transformer: Training

•
We can train transformer using cross entropy loss

•
We can understand the process of transformer training as a classification problem

𝑒𝑤1
𝑒𝑤2
𝑒𝑤𝑛

ℎ𝑤:1
ℎ𝑤:2
ℎ𝑤:𝑛
…

…
𝑒𝑤3

ℎ𝑤:3

𝑝𝑟𝑒𝑑𝑤:1
𝑝𝑟𝑒𝑑𝑤:2
𝑝𝑟𝑒𝑑𝑤:𝑛
…
𝑝𝑟𝑒𝑑𝑤:3

𝑤1
𝑤2
𝑤𝑛
…
𝑤3

𝑤2
𝑤3
𝑤𝑛+1
…
𝑤4
Cross Entropy Loss

Input

Output

## --- [Page 92] ---
92

Natural Language Generation with Transformer: 
Inference

•
The generation of text is performed by repeatedly predicting the next word

•
The predicted word is inserted at the end of the sentence and used as the input of the GPT model

무엇보다
Next word prediction


|  |  |  | ormer |  |  |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
| 근육이 _ | 커 _ | 지기 |  | 위해서 _ | 는 |

## --- [Page 93] ---
93

Natural Language Generation with Transformer: 
Inference

무엇보다

규칙적인

Transformer
Transformer

•
The generation of text is performed by repeatedly predicting the next word

•
The predicted word is inserted at the end of the sentence and used as the input of the GPT model


| 근육이 _ | 커 _ | 지기 | 위해서는 _ |  |
| --- | --- | --- | --- | --- |


| 근육이 | 커 | 지기 | 위해서 _ 는 | 무엇보다 |  |
| --- | --- | --- | --- | --- | --- |
| _ | _ |  |  |  |  |
|  |  |  |  |  |  |

## --- [Page 94] ---
94

Natural Language Generation: Search Algorithm

Greedy Search
Beam Search

•
Greedy search just selects the word with the highest probability

•
Beam search keeps the most likely num_beams of hypotheses at each time step and eventually
choosing the highest probability path

## --- [Page 95] ---
95

Summary

•
We trace the evolution from RNN-based models to the Transformer, driven by the need to solve long-range dependencies and

enable parallel processing.

•
Seq2Seq (Sequence-to-Sequence) models use an Encoder RNN to compress an entire input sequence into a single "context

vector," and a Decoder RNN to generate an output sequence from that vector.

•
The information bottleneck of this single vector was solved by the Attention Mechanism, allowing the Decoder to "look back"

at all input hidden states and assign "attention scores" to focus on the most relevant parts of the input for each step.

•
Despite this improvement, RNNs remain slow due to their sequential nature, leading to the Transformer architecture, which

removes RNNs entirely.

•
The Transformer's core component is Self-Attention, a mechanism that allows each word in a sequence to look at and weigh

the importance of all other words in the same sequence, creating a deeply context-aware representation.

•
The full Transformer model is an Encoder-Decoder stack built from Multi-Head Attention (running self-attention in parallel) and

Feed-Forward Networks, using Positional Embeddings to retain word order information.

•
During inference, this model generates text autoregressively by predicting one word at a time, feeding that prediction back into

the input, and then predicting the next word, often using a Beam Search algorithm to find the most probable sequence.

## --- [Page 96] ---
LG AI Research Special Lecture

Lecture 04 – 거대 언어 모델의 사전학습과 진화

중앙대학교 AI학과

이환희

딥러닝 자연어처리 기초와 LLM 에이전트


|  |  |
| --- | --- |
|  | 96 |

## --- [Page 97] ---
97

Outline

•
Pre-trained Word Embeddings

•
Pre-training Whole Models

•
In-context Learning

•
Reinforcement Learning from Human Feed back

•
InstructGPT

•
ChatGPT

거대 언어 모델의 사전학습과 진화

## --- [Page 98] ---
98

History of Large Language Models

Phase 1
Phase 2
Phase 3

GPT-4

Slides Adapted from Hui Yang

Gemini
Claude-3

2024
2025

Deepseek-r1

GPT-o3

## --- [Page 99] ---
99

NLP Milestone Papers

Slides Adapted from Hui Yang

• Word2Vec: Mikolov et al. "Distributed representations of words and phrases and their compositionality." NeurIPS 2013

• Attention: Bahdanau et al. "Neural machine translation by jointly learning to align and translate." ICLR 2015

• Transformer: Vaswani et al. "Attention is all you need.“ NIPS 2017.

• BERT: Devlin et al. "Bert: Pre training of deep bidirectional transformers for language understanding." NAACL 2019

• GPT-1: Radford et al. "Improving language understanding by generative pre training." (2018).

• GPT-2: Radford et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.

• BART: Lewis et al. "Bart: Denoising sequence to sequence pre training for natural language generation, translation, and comprehension."

ACL 2020

• GPT-3: Brown et al. "Language models are few shot learners, NeurIPS 2020

• InstructGPT: Ouyang et al. "Training language models to follow instructions with human feedback.“, arXiv

• Chain Of Thought: Wei et al. "Chain of thought prompting elicits reasoning in large language models."arXiv:2201.11903

• RLHF (Anthropic): Bai et al. "Training a helpful and harmless assistant with reinforcement learning from human feedback ." arXiv:2204.05862

• ChatGPT: No paper, still secrets from OpenAI

• LLaMA: Touvron et al. "LLaMA: Open and Efficient Foundation Language Models."arXiv:2302.13971

• GPT-4: GPT-4 Technical Report

• DeepSeek-R1: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, arXiv:2501

Pre-training LM

Pre-training LLM

## --- [Page 100] ---
100

Pre-trained Word Embeddings

•
Start with pre-trained word embeddings (no con
text!)

•
Learn how to incorporate context in an LSTM or
Transformer while training on the task.

Some issues to think about:

•
The training data we have for our downstream task (
like question answering) must be sufficient to teach 
all contextual aspects of language.

•
Most of the parameters in our network are randoml
y initialized!

[Recall, movie gets the same word embedding,

no matter what sentence it shows up in]

Slides Adapted from CS224n

## --- [Page 101] ---
101

Pre-trained Whole Models

In modern NLP:

•
All (or almost all) parameters in NLP networks are initi
alized via pretraining.

•
Pretraining methods hide parts of the input from the
model, and train the model to reconstruct those parts
.

•
This has been exceptionally effective at building
strong:

•
representations of language
•
parameter initializations for strong NLP models.
•
Probability distributions over language that we c
an sample from

Slides Adapted from CS224n

## --- [Page 102] ---
102

Pre-training for Three Types of Architectures

Slides Adapted from CS224n

•
Language models! What we’ve seen so far.

•
Nice to generate from; can’t condition on future words

•
Gets bidirectional context – can condition on future!

•
How do we train them to build strong representations?

•
Good parts of decoders and encoders?

•
What’s the best way to pretrain them?

•
The neural architecture influences the type of pre-training, and natural use cases.

BERT

BART, T5

GPT

## --- [Page 103] ---
103

What Can we Learn from Reconstructing the Input?

There’s increasing evidence that pretrained models learn a wide variety of things about
the statistical properties of language.

•
Chung-Ang University is located in ___, Seoul. [Trivia]

•
I put __  fork down on the table. [syntax]

•
The woman walked across the street, checking for traffic over ___shoulder. [coreference]

•
I went to the ocean to see the fish, turtles, seals, and ___. [lexical semantics/topic]

•
Overall, the value I got from the two hours watching it was the sum total of the popcorn and the drink. The movie was ____.

[sentiment]

•
Iroh went into the kitchen to make some tea. Standing next to Iroh, Zuko pondered his destiny. Zuko left the ____.

[some reasoning – this is harder]

•
I was thinking about the sequence that goes 1, 1, 2, 3, 5, 8, 13, 21,____

[some basic arithmetic; they don’t learn the Fibonnaci sequence]

Slides Adapted from CS224n

## --- [Page 104] ---
104

•
Model 𝑝𝜃 
𝑤𝑡𝑤1:𝑡−1), the probability 
distribution over words given their pas
t contexts.
•
There’s lots of data for this!

Pre-training through language modeling

Recall the language modeling task:

Pretraining through language modeling:

•
Train a neural network to perform language 
modeling on a large amount of text.
•
Save the network parameters.

Decoder (Transformer, LSTM, ++ )

Iroh
goes
to
make
tasty
tea

goes
to
make
tasty
tea
END

Slides Adapted from CS224n
Dai et al., Semi-supervised Sequence Learning, NeurIPS 2015

## --- [Page 105] ---
105

Pre-training / Fine-tuning Paradigm

•
Pretraining can improve NLP applications by serving as parameter initialization.

(Transformer, LSTM, ++ )

Iroh
goes
to
make
tasty
tea

goes
to
make
tasty
tea
END

Step 1: Pretrain (on language modeling)

Lots of text; learn general things!

Step 2: Finetune (on your task)
Not many labels; adapt to the task!

(Transformer, LSTM, ++ )

☺/

… the movie was …

Slides Adapted from CS224n

## --- [Page 106] ---
106

Pre-training / Fine-tuning Paradigm

*Model

*Dataset

BooksCorpus
(800M words)

*Objective

Wikipedia
(2,500M words)

(1) Predict the masked word
(2) Next sentence prediction

BERT

*Model
Classifier

BERT
(Pre-trained)

75%

25%

Spam

Not Spam

*Dataset

Email content
Label

(1) Pre-training (self-supervised)

Training on large amounts of data
(Language modeling)

(2) Fine-tuning (supervised)

Training on a specific downstream task
with a labeled dataset

Justone additionaloutput layer


| Buy one, get one free | Spam |
| --- | --- |
| Dear Harry, Hi this is.. | Not Spam |

## --- [Page 107] ---
107

Pre-training “Large” Language Models

• Pre-training a very large model using massive amounts of data on the web by predicting the next 
word.

• Looks like reading all the documents in the world as if reading books.

• Through this process, various "common sense" and "prior knowledge" are acquired.

## --- [Page 108] ---
108

GPT-3: Intialization of Large-scale “Pre-training”

• Large-scale language model pre-training

▪Published in June 2020

▪Architecture: 175 Billion parameter Transformer (96 layers, 12k hidden dim, 96 heads)

▪Dense attentions & sparse attentions used

▪Dataset: 300 Billion tokens (60% CommonCrawl + 22% WebText2 + 16% Books + 3% Wikipedia)

▪Model not available, Commercial API available

## --- [Page 109] ---
109

Pre-training LLM GPT-3 and In-Context Learning

• In-context Learning – kind of Few-shot Learning

1) Zero-shot

2) Few-shot

Brown et al., Language Models are Few-Shot Learners, NeurIPS 2020

## --- [Page 110] ---
110

Advantages of GPT-3: In-Context Learning

• Versatility: One model can solve multiple problems

•
Summarization, Programming, Translation, Sentiment Analysis, etc.

• No need for large amounts of human-labeled learning data for fine-tuning

• Allows to solve new problems that are not solved due to the lack of dataset

## --- [Page 111] ---
111

Limitation of GPT-3 & Instruct GPT

https://openai.com/research/instruction-following

•
GPT-3 generates the next words/sentences well

•
GPT-3 cannot generate sentences well for a given instruction

•
InstructGPT: Language model more aligned with their users (sibling models 
for ChatGPT)

## --- [Page 112] ---
112

Challenges in Developing better Language Model

“The three Hʼs of Model Desiderata”

1) Helpful:

•
The AI should help the user solve their task

2) Honest:

•
The AI should give accurate information

3) Harmless:

•
The AI should not cause physical, psychological, or social harm to people or the 
environment

Ouyang et al., Training language models to follow instructions with human feedback

## --- [Page 113] ---
113

Challenges in Developing better Language Model

• Misalignment: When the training objective does not capture the desiderata we want from models

Ouyang et al., Training language models to follow instructions with human feedback

Training: Next-token prediction

Evaluation: Follow instructions (e.g. summarize this)

“Misalignment”

## --- [Page 114] ---
114

Solution: Human Feedback

• Solution to misalignment: learn directly from human feedback

Ouyang et al., Training language models to follow instructions with human feedback

## --- [Page 115] ---
115

Training LMs to Follow Instructions with Human 
Feedback

1. Supervised Fine-Tuning (SFT)

• Using human-written demonstrations

2. Reinforcement Learning from Human Feedback (RLHF)

• Agent: Language Model

• Environment: Human users

• State: Human Inputs

• Action: Model Output

• Policy: Language Model Generation given Input

• Reward: Human Feedback

## --- [Page 116] ---
116

Supervised Fine-tuning (SFT)

•
Collect demonstrations for prompt from humans

•
Train the model to generate the demonstrations from the prompt

Ouyang et al., Training language models to follow instructions with human feedback

1) Sample the prompt
2) Demonstrate the desired 
output behavior by labelers
3) Finetune GPT-3 model

## --- [Page 117] ---
117

Supervised Fine-tuning (SFT)

• Prompt example and statistics

Ouyang et al., Training language models to follow instructions with human feedback

## --- [Page 118] ---
118

Reinforcement Learning from Human Feedback (RLHF)

① Collect comparison data and train a reward model

https://openai.com/blog/chatgpt

1) Collect a prompt and
model outputs

2) Rank the outputs by 
humans

3) Train reward model

## --- [Page 119] ---
119

Reinforcement Learning from Human Feedback (RLHF)

https://openai.com/blog/chatgpt

2) Generate an 
output

② Optimize a policy against the reward model using reinforcement learning

3) Compute a reward 
for the output by 
reward model

4) Update the model 
with the reward

Repeat 1) to 4)

1) Sample a 
new prompt

•
During training, following loss function is used

makes sure PPO model 
output does not deviate 
too far from SFT

auxiliary LM objective on 
the pre-training data

## --- [Page 120] ---
120

Why is RLHF Good Compared to just SFT?

1) Reward is a more nuanced training signal than autoregressive loss

- If the correct next token is “great”, the AR loss penalizes the prediction “amazing” the same as “sandwiches”. 
The RM assigns similar rewards to sequences with similar quality

2) The RM “critiques” actual completions generated from the model itself, whereas 
SFT training does not use model generations, since it is completely offline.

- This means the RM may provide more “tailored” feedback to the model

## --- [Page 121] ---
121

Why is RLHF Good Compared to just SFT?

3) The RM more directly captures the notion of “preference”

- Preferences induce rankings, and rankings can be used to infer preferences

- Ranking is very naturally captured by the reward signal, better sequences = higher reward

- In SFT, preference is not explicitly captured, since we only train to regurgitate “the best” example

4) The RM is more data efficient

- There is a reason step 1 uses 13k prompts, but step 3 can use 31k prompts.

- For SFT, we need humans to generate target. Once we train the RM, it can be used to score any output

## --- [Page 122] ---
122

InstructGPT: Human Preference Test

Ouyang et al., Training language models to follow instructions with human feedback

• GPT-3 + SFT + RLHF (InstructGPT) shows best results!

## --- [Page 123] ---
123

InstructGPT: Preference Breakdown

Ouyang et al., Training language models to follow instructions with human feedback

•
InstructGPT models follow the instruction more correctly (helpful)

•
InstructGPT models less likely to hallucinate (honest)

## --- [Page 124] ---
124

From InstructGPT to ChatGPT

• “…same method as InstructGPT, but with slight differences in the data collection 
setup”

▪
Similar to SFT model of InstructGPT

▪
Aggregate NLP datasets, write prompt templates, and fine-tune model

• SFT data

▪human AI trainers provided conversations in which they played both sides—the user and an AI assistant

▪mixed this new dialogue dataset with the InstructGPT dataset, which we transformed into a dialogue format

• Reward Model data

▪took conversations that AI trainers had with the chatbot. We randomly selected a model-written message,

▪sampled several alternative completions, and had AI trainers rank them

## --- [Page 125] ---
125

Limitation of Naïve ChatGPT: Lack of up-to-date 
Information

## --- [Page 126] ---
126

Retrieval Augmented Generation: Accurate Response 
using Search Engine

https://lemaoliu.github.io/retrieval-generation-tutorial/assets/slides/retrieval4lm.pdf

## --- [Page 127] ---
127

Downloading Pre-trained Models: Instruct Model vs Base 
Model

•
A Base Model is the result of the initial pre-training phase, excelling at text completion.

•
An Instruct Model is created by fine-tuning a Base Model to align with user intent, making it optimized

for answering questions and following instructions.

## --- [Page 128] ---
128

Training Paradigm in NLP

Supervised Learning (~2019)

Pre-training + Fine-tuning (2019~)

Pre-training + Few-Shot or Zero-Shot or Parameter

Efficient Tuning (2020~)

Pre-training + Instruction Tuning + RLHF (2022~)
GPT-3.5, GPT-4
(ChatGPT, Instruct GPT)

GPT-3

BERT,

GPT, 
BART
Model Size↑, 
Pre-training Dataset

Size ↑

## --- [Page 129] ---
129

Evolutionary Tree of Modern LLMs until 2023

Yang et al., Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond, arXiv

## --- [Page 130] ---
130

Summary

•
We trace the evolution of pre-training, from learning static Word Embeddings to Pre-training Whole Models like BERT 
and GPT, which learn deep contextual representations from massive, unlabeled text.

•
This established the dominant Pre-training / Fine-tuning paradigm, where a model is first trained on a general, self-
supervised task and then adapted with a small amount of labeled data for a specific downstream task.

•
The scaling of this approach led to Large Language Models (LLMs) like GPT-3, which demonstrated In-context Learning
(zero-shot and few-shot), allowing the model to perform new tasks from just a prompt without any gradient updates.

•
However, this scaling also revealed a "Misalignment" problem: a model trained to predict the next word isn't necessarily 
helpful, honest, or harmless.

•
InstructGPT aligns models to user intent through two-stage: first, Supervised Fine-Tuning (SFT) on human-written 
examples, and second, Reinforcement Learning from Human Feedback (RLHF) to "reward" outputs that humans prefer.

•
ChatGPT was then developed by applying a similar instruction-tuning and RLHF methodology, specializing the model for 
a dialogue format.

•
Finally, to overcome the static knowledge of pre-trained models, Retrieval-Augmented Generation (RAG) is used to 
provide the model with up-to-date, external information (e.g., from a search) before it generates an answer.

## --- [Page 131] ---
LG AI Research Special Lecture

Lecture 05 – 스스로 계획하고 실행하는 AI, LLM 에이전트

중앙대학교 AI학과

이환희

딥러닝 자연어처리 기초와 LLM 에이전트


|  |  |
| --- | --- |
|  | 131 |

## --- [Page 132] ---
132

Outline

•
Definition of LLM Agents

•
Memory of LLM

•
Using Tools for LLM & MCP

•
Planning, ReAct

•
Multi-agent Collaboration

스스로 계획하고 실행하는 AI, LLM 에이전트

## --- [Page 133] ---
133

What are LLM Agents?

•
There are many other tasks that LLMs often fail at, including basic math like multiplication and division

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 134] ---
134

What are LLM Agents?

•
Through external systems, the capabilities of the LLM can be enhanced. Anthropic calls this “The Augmented LLM”.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 135] ---
135

What are LLM Agents?

•
For instance, when faced with a math question, the LLM may decide to use the appropriate tool (a calculator).

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 136] ---
136

What are LLM Agents?

•
An agent is anything that can be viewed as perceiving its environment through sensors and acting upon that 
environment through actuators.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

Environments — The world the 
agent interacts with

Sensors — Used to observe the 
environment

Actuators — Tools used to interact 
with the environment

Effectors — The “brain” or rules 
deciding how to go from 
observations to actions

## --- [Page 137] ---
137

What are LLM Agents?

•
We can generalize this framework a bit to make it suitable for the “Augmented LLM” as below.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

Environments — The world the 
agent interacts with

Sensors — Used to observe the 
environment

Actuators — Tools used to interact 
with the environment

Effectors — The “brain” or rules 
deciding how to go from 
observations to actions

## --- [Page 138] ---
138

What are LLM Agents?

•
Using the reasoning behavior, LLM Agents will plan out the necessary actions to take.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 139] ---
139

What are LLM Agents?

•
This planning behavior allows the Agent to understand the situation (LLM), plan next steps (planning), take 
actions (tools), and keep track of the taken actions (memory).

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 140] ---
140

What are LLM Agents?

•
Depending on the system, LLM Agents have varying degrees of autonomy.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 141] ---
141

Short-term Memory

•
LLMs are forgetful systems, or more accurately, do not perform any memorization at all when interacting with them.

•
When you ask an LLM a question and then follow it up with another question, it will not remember the former.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 142] ---
142

Long-term Memory

•
LLM Agent also needs to keep track of potentially dozens of steps, not only the most recent actions.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 143] ---
143

Long-term Memory

•
This is referred to as long-term memory as the LLM Agent could theoretically take dozens or even hundreds of 
steps that need to be memorized.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 144] ---
144

Implementing Short-term Memory

•
Model's context window enables short-term memory

•
This works as long as the conversation history fits within the LLM’s context window.

•
However, instead of actually memorizing a conversation, wae essentially “tell” the LLM what that conversation was.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 145] ---
145

Implementing Short-term Memory

•
For models with a smaller context window, or when the conversation history is large, we can instead use another 
LLM to summarize the conversations that happened thus far.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 146] ---
146

Implementing Long-term Memory

•
A common technique to enable long-term memory is to store all previous interactions, actions, and conversations in

an external vector database.

•
To build such a database, conversations are first embedded into numerical representations that capture their meaning.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 147] ---
147

Implementing Long-term Memory

•
After building the database, we can embed any given prompt and find the most relevant information in the

vector database by comparing the prompt embedding with the database embeddings.

•
This method is often referred to as Retrieval-Augmented Generation (RAG).

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 148] ---
148

Memory Types

•
Different types of information can also be related to different types of memory to be stored. 
•
In psychology, there are numerous types of memory to differentiate, but the Cognitive Architectures for 
Language Agents paper coupled four of them to LLM Agents.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

Semantic memory (facts about the world) might be stored in a different database 
than working memory (current and recent circumstances).

## --- [Page 149] ---
149

Using Tools to LLM

•
Tools allow a given LLM to either interact with an external environment (such as databases) or use external 
applications (such as custom code to run).
•
Tools generally have two use cases: fetching data to retrieve up-to-date information and taking action like 
setting a meeting or ordering food.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 150] ---
150

Using Tools to LLM

•
Function calling: generate custom functions that the LLM can use, like a basic multiplication function.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

•
Some LLMs can use any tools if they are prompted correctly and extensively. Tool-use is something that most 
current LLMs are capable of.

## --- [Page 151] ---
151

Using Tools to LLM

•
Tools can either be used in a given order if the agentic framework is fixed…

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 152] ---
152

Using Tools to LLM

•
LLM can autonomously choose which tool to use and when.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 153] ---
153

Using Tools to LLM

•
In other words, the output of intermediate steps is fed back into the LLM to continue processing.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 154] ---
154

Toolformer

•
Tool use is a powerful technique for strengthening LLMs' capabilities and compensating for their 
disadvantages. As such, research efforts on tool use and learning have seen a rapid surge in the last few years.
•
Toolformer is a model trained to decide which APIs to call and how.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 155] ---
155

Toolformer

•
[ and ] tokens to indicate the start and end of calling a tool.
•
LLM generates tokens until it reaches the → token which indicates that the LLM stops generating tokens.
•
Then, the tool will be called, and the output will be added to the tokens generated thus far.
•
The ] symbol indicates that the LLM can now continue generating if necessary.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 156] ---
156

Toolformer

•
Toolformer creates this behavior by carefully generating a dataset with many tool uses the model can train on. 
For each tool, a few-shot prompt is manually created and used to sample outputs that use these tools.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 157] ---
157

Model Context Protocol (MCP)

•
Tools are an important component of Agentic frameworks, allowing LLMs to interact with the world and 
extend their capabilities. 
•
However, enabling tool use when you have many different API becomes troublesome as any tool needs to be:

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

▪Manually tracked and fed to the LLM

▪Manually described (including its

expected JSON schema)

▪Manually updated whenever its API

changes

## --- [Page 158] ---
158

Model Context Protocol (MCP)

•
MCP Host — LLM 
application (such as Cursor) 
that manages connections

•
MCP Client — Maintains 
1:1 connections with MCP 
servers

•
MCP Server — Provides 
context, tools, and 
capabilities to the LLMs

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

•
For easier implementation any given Agentic framework, Anthropic developed the Model Context Protocol (MCP). 
•
MCP standardizes API access for services like weather apps and GitHub.

Three components of MCP

## --- [Page 159] ---
159

Model Context Protocol (MCP)

•
Let’s assume you want a given LLM application to summarize the 5 latest commits from your repository.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents
•
The MCP Host (together with the client) would first call the MCP Server to ask which tools are available.

## --- [Page 160] ---
160

Model Context Protocol (MCP)

•
Let’s assume you want a given LLM application to summarize the 5 latest commits from your repository.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

•
The LLM receives the information and may choose to use a tool. It sends a request to the MCP Server via the 
Host, then receives the results, including the tool used.

## --- [Page 161] ---
161

Model Context Protocol (MCP)

•
Let’s assume you want a given LLM application to summarize the 5 latest commits from your repository.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

•
Finally, the LLM receives the results and can parse an answer to the user.

## --- [Page 162] ---
162

Planning

•
Tool use allows an LLM to increase its capabilities. They are typically called using JSON-like requests.

•
But how does the LLM, in an agentic system, decide which tool to use and when?

•
This is where planning comes in. Planning in LLM Agents involves breaking a given task up into actionable steps.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 163] ---
163

Planning

•
Plan allows the model to iteratively reflect on past behavior and update the current plan if necessary.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

•
To enable planning in LLM Agents, let’s first look at the foundation of this technique, namely reasoning.

## --- [Page 164] ---
164

Reasoning

•
Planning actionable steps requires complex reasoning behavior. As such, the LLM must be able to showcase

this behavior before taking the next step in planning out the task.

•
“Reasoning” LLMs are those that tend to “think” before answering a question.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 165] ---
165

Reasoning

•
Enabling reasoning behavior in LLMs is great but does not necessarily make it capable of planning actionable steps.

•
Chain-of-Thought, for instance, is focused purely on reasoning.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 166] ---
166

ReAct: Reason and Act

•
One of the first techniques to combine both processes is called ReAct (Reason and Act)

•
ReAct does so through careful prompt engineering.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 167] ---
167

ReAct: Reason and Act

•
The ReAct prompt describes three steps:

•
Thought - A reasoning step about the current situation
•
Action - A set of actions to execute (e.g., tools)
•
Observation - A reasoning step about the result of the action

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 168] ---
168

ReAct: Reason and Act

•
The LLM uses this prompt (which can be used as a system prompt) to steer its behaviors to work in cycles of

thoughts, actions, and observations.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 169] ---
169

Reflecting

•
Nobody, not even LLMs with ReAct, will perform every task perfectly. Failing is part of the process as long as 
you can reflect on that process.

•
This process is missing from ReAct and is where Reflexion comes in. Reflexion is a technique that uses verbal 
reinforcement to help agents learn from prior failures.

The method assumes three LLM roles:

Actor — Chooses and executes 
actions based on state observations. 
We can use methods like Chain-of-
Thought or ReAct.

Evaluator — Scores the outputs 
produced by the Actor.

Self-reflection — Reflects on the 
action taken by the Actor and scores 
generated by the Evaluator.
Memory modules are added to track actions and self-reflections.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 170] ---
170

Multi-Agent Collaboration

•
We can look towards Multi-Agents, frameworks where multiple agents (each with access to tools, memory, and

planning) are interacting with each other and their environments.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 171] ---
171

Multi-Agent Collaboration

•
These Multi-Agent systems usually consist of specialized Agents, each equipped with their own toolset and overseen

by a supervisor. The supervisor manages communication between Agents and can assign specific tasks to the

specialized Agents.

•
Each Agent might have different types of tools available, but there might also be different memory systems.

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 172] ---
172

Multi-Agent Collaboration

•
In practice, there are dozens of Multi-Agent architectures with two components at their core:

•
Agent Initialization — How are individual (specialized) Agents created?

•
Agent Orchestration — How are all Agents coordinated?

https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents

## --- [Page 173] ---
173

Summary

•
LLM Agent Definition: An "Augmented LLM" that acts as the "brain" (or effector) in a classic agent framework. It is given access to

external Tools (actuators), a Memory module, and a Planning component to autonomously solve complex tasks.

•
Short-Term Memory: This is how an agent remembers the current conversation. It's typically implemented by fitting the entire

conversation and action history into the LLM's context window.

•
Long-Term Memory: To remember information across sessions, agents use an external Vector Database. Past conversations are stored

as embeddings, and the agent uses RAG (Retrieval-Augmented Generation) to retrieve this relevant information when needed.

•
Tools & Function Calling: Tools are external APIs (e.g., search(), run_code(), calculator()) that allow the agent to interact with the world.

The LLM is trained to output structured JSON (Function Calling) to autonomously decide which tool to use and when.

•
Planning: This is the agent's ability to decompose a complex user request into a series of smaller, actionable steps. This requires the LLM

to have "reasoning" capabilities.

•
ReAct (Reason + Act): The key technique that combines reasoning and action. The LLM is prompted to repeat a cycle: 1. Thought

(private reasoning about what to do next), 2. Action (choosing and calling a tool), and 3. Observation (analyzing the tool's output to

inform the next thought).

•
Advanced Agent Frameworks: This concept can be extended with Reflexion (using an "Evaluator" LLM to learn from failures) or Multi-

Agent Collaboration, where a "Supervisor" agent delegates tasks to a team of specialized agents to solve the problem.

## --- [Page 174] ---
LG AI Research Special Lecture

Lecture 06 – LLM을 움직이는 힘, AI 하드웨어와 GPU

중앙대학교 AI학과

이환희

딥러닝 자연어처리 기초와 LLM 에이전트


|  |  |
| --- | --- |
|  | 174 |

## --- [Page 175] ---
175

Outline

• Compute Internals

• CPU Internals

• GPU Computing

• CPU vs GPU

• Floating-Point Number

• Size Guide for LLM Models & GPU VRAM

LLM을 움직이는 힘, AI 하드웨어와 GPU

## --- [Page 176] ---
176

These Photos by Unknown Author is licensed under CC BY-SA

Computer Internal

## --- [Page 177] ---
177

Operating System (OS)

• Layer between Hardware and Software

Justice, Matthew. How Computers Really Work (p. 196). No Starch Press. Kindle Edition.

## --- [Page 178] ---
178

Operating System (OS)

• An operating system is responsible for

•
Resource management

•
Process Management

•
Facilitating device I/O

•
Providing a set of system services for applications.

•
User Interface

•
Security and Permission Management

## --- [Page 179] ---
179

Operating System Kernel

• The operating system kernel allows multiple programs to run in parallel and share hardware

resources. Kernel is the core part of an operating system, but it alone provides no way for end users to

interact with the system.

• Operating systems also include non-kernel components that are needed for a system to be of use.

This includes the shell, a user interface for working with the kernel. The terms shell and kernel are

part of a metaphor for operating systems, where the OS is thought of as a nut or seed. The kernel is at

the core; a shell surrounds it.

Justice, Matthew. How Computers Really Work (pp. 195-196). No Starch Press. Kindle Edition.

## --- [Page 180] ---
180

Hardware Interface

•
The shell can be either a command line interface (CLI) or a graphical user interface (GUI). Some examples

of shells are the Windows shell GUI (including the desktop, Start menu, taskbar, and File Explorer), and the

Bash shell CLI found on Linux and Unix systems.

•
When it comes to interacting with hardware, the kernel acts in partnership with device drivers. A device

driver, or simply driver, is software designed to interact with specific hardware. Operating systems typically

include a set of device drivers for common hardware and also provide a mechanism for installing additional

drivers.

Justice, Matthew. How Computers Really Work (p. 196). No Starch Press. Kindle Edition.

## --- [Page 181] ---
181

Input/Output Operations

• Computers need to interact with external environment

•
I/O Devices

•
OS has software device drivers for each device

## --- [Page 182] ---
182

Processors

• CPU – Central Processing Units

• GPU – Graphics Processing Units

• FPGA – Field Programmable Gate Arrays

• ASIC – Application Specific Integrated Circuits

Retrieved from AI hardware - What they are and why they matter in 2021 (roboticsbiz.com)

## --- [Page 183] ---
183

What is CPU?


| A | “ | central processing unit | “ |  |
| --- | --- | --- | --- | --- |


## --- [Page 184] ---
184

Program execution in CPU

• In serial processing, the processor executes the program instructions sequentially, one after the other. 
After completing that, it executes the next instruction in a sequential manner.

• CPU follows the steps below to process a sequence of program instructions that stored in main 
memory(RAM).

1- Fetch the instruction

2- Decode the instruction

3- Execute the instruction

## --- [Page 185] ---
185

CPU Cores

• PHYSICAL AND LOGICAL CORES - Not all cores are equally capable of parallelism. A physical core 
is a hardware implementation of a core within a CPU. Logical cores represent the ability of a 
single physical core to run multiple threads at once (one thread per logical core). Intel refers to 
this capability as hyper-threading.

• A computer with two physical cores, each with two logical cores, has a total of four logical cores 
and run four threads at once. Logical cores cannot achieve the full parallelism of physical cores.

Justice, Matthew. How Computers Really Work (p. 203). No Starch Press. Kindle Edition.

## --- [Page 186] ---
186

CPU

• Latency – beginning to end duration of performing a single computation (*Tuomanen, 2018)

• CPUs are engineered to reduce the latency of a single computation.

• Computations are sequential

• Fewer cores that are restricted with the number of processes it can compute and it handles

those processes very fast

*Hands-On GPU Programming with Python and CUDA

## --- [Page 187] ---
187

What is GPU?

• GPU stands for “Graphic Processing Unit” that is a piece of hardware designed to accelerate graphics

rendering tasks.

engineering computing.


| GPU computing is the use of the GPU as a co | - | processor to accelerate CPUs for general | - | purpose scientific and |
| --- | --- | --- | --- | --- |


| It can be used to execute |  | the instructions |  | parallelly on multiple GPU cores, thus accelerating the processing. |
| --- | --- | --- | --- | --- |


## --- [Page 188] ---
188

GPU

• Each core is much simpler than that of the CPU and each core on its own is not as fast as a CPU

• It’s the number of cores that make a difference. GPU has hundreds to thousands of cores 
compared to a CPU which usually has 1 to 6 cores

• Computations are done in parallel, asynchronously.

• Still relies on CPU for managing and passing data

• Programs must be rewritten to enable parallel processing

## --- [Page 189] ---
189

Differences CPU and GPU


| CPU | GPU |
| --- | --- |
| Several cores | Many cores |
| Designed to minimize latency | Designed to maximize throughput |
| Good for serial processing | Good for parallel processing |
| Multiple type of data | Single type of data |
| Can do a handful of operations at once | Can do thousands of operations at once |

## --- [Page 190] ---
190

Accelerated Computing
10x Performance & 5x Energy Efficiency for HPC

CPU
Optimized for

Serial Tasks

GPU Accelerator

Optimized for 
Parallel Tasks
CPU Strengths

•
Very large main memory
•
Very fast clock speeds
•
Latency optimized via large caches
•
Small number of threads can run very 
quickly

CPU Weaknesses

•
Relatively low memory bandwidth
•
Cache misses very costly
•
Low performance/watt

© NVIDIA 2013
Slide adapted from NVIDIA Accelerated Computing Teaching Kit

## --- [Page 191] ---
191

Accelerated Computing
10x Performance & 5x Energy Efficiency for HPC

CPU
Optimized for

Serial Tasks

GPU Accelerator

Optimized for 
Parallel Tasks

GPU Strengths

•
High bandwidth main memory
•
Significantly more compute resources
•
Latency tolerant via parallelism
•
High throughput
•
High performance/watt

GPU Weaknesses

•
Relatively low memory capacity
•
Low per-thread performance

© NVIDIA 2013
Slide adapted from NVIDIA Accelerated Computing Teaching Kit

Click to add text

## --- [Page 192] ---
192

Accelerator Nodes

PCIe

RAM
RAM

CPU and GPU have distinct 
memories

•
CPU generally larger 
and slower

•
GPU generally smaller 
and faster

CPU and GPU 
communicate via PCIe

•
Data must be copied 
between these 
memories over PCIe

•
PCIe Bandwidth is much 
lower than either 
memories

Emerging Tech - Nvlink

© NVIDIA 2013
Slide adapted from NVIDIA Accelerated Computing Teaching Kit

## --- [Page 193] ---
193

Heterogeneous Programming

Application Code

+

GPU
CPU

A few % of Code
A large % of Time

Compute-Intensive

Functions

Rest of Sequential

CPU Code

© NVIDIA 2013
Slide adapted from NVIDIA Accelerated Computing Teaching Kit

## --- [Page 194] ---
194

Low Bit-Width Operations are Cheap

• Less Bit-Width → Less Energy

Slides adapted from Han’s 6.5940

## --- [Page 195] ---
195

Data Types: Integer

• Unsigned Integer

• n-bit Range:[0, 2^n − 1]

• Signed Integer

• Sign-Magnitude Representation

• n-bit Range: [−2^n−1 − 1, 2^n−1 − 1]

• Both 000…00 and 100…00 represent 0

• Two’s Complement Representation

• n-bit Range: [−2^n−1, 2^n−1 − 1]

• 000…00 represents 0

• 100…00 represents −2n−1

Slides adapted from Han’s 6.5940

+

## --- [Page 196] ---
196

Fixed-Point Number

Slides adapted from Han’s 6.5940

## --- [Page 197] ---
197

Floating-Point Number

• Example: 32-bit floating-point number in IEEE 754

Slides adapted from Han’s 6.5940

## --- [Page 198] ---
198

Floating-Point Number

• Exponent Width -> Range; Fraction Width -> Precision

Slides adapted from Han’s 6.5940

## --- [Page 199] ---
199

Numeric Data Types

• Question: What is the following IEEE half precision (IEEE FP16) number in decimal?

Slides adapted from Han’s 6.5940

## --- [Page 200] ---
200

LLM Models & GPU VRAM: A Sizing Guide

• An analysis of VRAM requirements based on model size (parameters), precision, and context length.

• To run an LLM (for inference or training), all the necessary data must be loaded into the GPU's high-speed 
VRAM. If VRAM is insufficient, the model cannot run or must "offload" data to slower system RAM, 
dramatically reducing speed.

• Key VRAM Consumers:

• Model Weights: The single largest component.

• Optimizer States: (During full fine-tuning) Can be 2x or more the size of the weights (e.g., using AdamW).

• Activations & Gradients: Intermediate calculations during the forward/backward pass.

• KV Cache: (Crucial for Inference). This stores the state of the sequence. It grows linearly with the context 
size (sequence length) and batch size.

## --- [Page 201] ---
201

Model Precision

• The VRAM footprint is determined by the precision used to store each parameter.

• FP32 (Full Precision): 1 parameter = 4 bytes

• e.g., Llama 3 8B (FP32) = 8B * 4 bytes ≈ 32 GB

• FP16 / BF16 (Half Precision): 1 parameter = 2 bytes (Today's Standard)

• e.g., Llama 3 8B (FP16) = 8B * 2 bytes ≈ 16 GB

• INT8 (Quantized): 1 parameter = 1 byte

• e.g., Llama 3 8B (INT8) = 8B * 1 byte ≈ 8 GB

• INT4 (e.g., QLoRA / GPT-Q): 1 parameter = 0.5 bytes (Heavily Quantized)

• e.g., Llama 3 8B (INT4) ≈ 4 GB (+ ~1-2GB overhead)

• Loading a model like Llama 3 8B in its standard FP16 precision requires at least 16GB of VRAM for the weights alone.

## --- [Page 202] ---
202

GPU Example : 16GB (e.g., Google Colab T4/L4, RTX 
4060Ti 16GB)

• Features: Highly accessible, common in cloud instances (Google Colab) and consumer gaming cards.

• Models (e.g., Mistral 7B, Llama 3 8B):

• Inference (FP16): Possible. (8B * 2 bytes ≈ 16GB). This is a perfect fit, but leaves little room for a long context 
(KV Cache).

• Full Fine-Tuning (FP16): Impossible. (16GB weights + 32GB+ optimizer + activations > 16GB).

• PEFT (LoRA / QLoRA): Possible. Loading the model in 4-bit (QLoRA, ≈ 5-6GB) leaves ample VRAM for fine-tuning.

• Context Impact: At 16GB, a long context (e.g., 32k tokens) during FP16 inference can easily cause an 
OOM (Out of Memory) error due to the large KV Cache.

## --- [Page 203] ---
203

VRAM for Full Fine-Tuning

• Training is the process of "updating the model's weights." It requires everything from inference, plus all the 
intermediate data needed to calculate the weight updates.

• Model Weights: (e.g., Llama 3 8B (FP16) ≈ 16GB)

• Gradients:

• These are the "instructions" for how much each weight needs to be updated.

• They require VRAM equal to the size of the model weights. (e.g., 16GB)

• Optimizer States:

• This is the main reason VRAM usage explodes.

• A standard AdamW optimizer requires 2x the size of the model weights. (e.g., 16GB * 2 = 32GB)

• Activations:

• Unlike inference, for backpropagation, the activations from all layers must be stored in VRAM throughout the 
training step. This can be a significant amount, growing with sequence length.

• Full-Tuning VRAM for 8B ≈ (Weights: 16GB) + (Gradients: 16GB) + (Optimizer States: 32GB) + (Activations: α) ≈ 64GB+

## --- [Page 204] ---
204

Summary

•
CPU, the computer's "brain," has a few powerful cores designed for serial tasks to minimize latency (the time for 
one operation), making it ideal for running the OS and sequential logic.

•
GPU is a massively parallel co-processor with thousands of simpler cores designed to maximize throughput (total 
operations at once), which is essential for the matrix math in deep learning.

•
In "Heterogeneous Programming," the CPU (host) runs the main application and offloads the small, compute-
intensive parts of the code to the GPU (accelerator) for parallel execution.

•
This is critical because LLMs rely on Floating-Point Numbers, and GPUs are optimized for these calculations; 
using lower precision (e.g., FP16 or INT8 instead of FP32) saves VRAM, energy, and time.

•
GPU VRAM is the primary bottleneck for running LLMs, as the model's Weights, Optimizer States (during 
training), and KV Cache (during inference) must all fit into this high-speed memory.

•
This has practical consequences: a Llama 3 8B model requires ~16GB of VRAM in standard FP16 precision, making 
it impossible to "full fine-tune" (which needs 64GB+) on a 16GB GPU, necessitating techniques like quantization 
(e.g., QLoRA).