## --- [Page 1] ---
Instructor: Jaehyung Kim

jaehyungk@yonsei.ac.kr

Decoding of Large Language Models

## --- [Page 2] ---
•Def. Language model (LM): probability distribution over sequences of tokens[1]

• 𝑥𝑖is a token with a finite vocabulary 𝑽
• E.g., 𝑝(𝗍𝗁𝖾,𝗆𝗈𝗎𝗌𝖾,𝖺𝗍𝖾,𝗍𝗁𝖾,𝖼𝗁𝖾𝖾𝗌𝖾)=0.02
• Autoregressive language models with chain rule are usually considered:

Language Models (LMs)

[1] https://stanford-cs324.github.io/winter2022/lectures/introduction/
1

## --- [Page 3] ---
•LM: “Transformer[1]” + “Self-supervised learning” +“Massive unlabeled text”

• E.g., Next token prediction[2] →learning knowledge from unlabeled text data

Recap: Language Models (LMs)

[1] Vaswani et al., Attention Is All You Need., NeurIPS 2017
[2] Brown et al., Language Models are Few-Shot Learners., NeurIPS 2020
2

## --- [Page 4] ---
•Def: mapping given input text to sequence of tokens

• There are 3 types: (1) character-level, (2) subword-level, and (3) word-level
• (Pre-defined) Set of all possible tokens: “Vocabulary”

Tokenization

3
[1] Source: https://medium.com/@abdallahashraf90x/tokenization-in-nlp-all-you-need-to-know-45c00cfa2df7

## --- [Page 5] ---
•Example: character-level tokenization
Tokenization

4
[1] Source: https://medium.com/@abdallahashraf90x/tokenization-in-nlp-all-you-need-to-know-45c00cfa2df7

## --- [Page 6] ---
•Example: character-level tokenization
Tokenization

5

Vocabulary (size 20)

[1] Source: https://medium.com/@abdallahashraf90x/tokenization-in-nlp-all-you-need-to-know-45c00cfa2df7

## --- [Page 7] ---
•Example: subword tokenization in practice (BERT-pretrained)
Tokenization

6
[1] Source: https://medium.com/@abdallahashraf90x/tokenization-in-nlp-all-you-need-to-know-45c00cfa2df7

## --- [Page 8] ---
•Def: mapping given input text to sequence of tokens

• There are 3 types: (1) character-level, (2) subword-level, and (3) word-level
• (Pre-defined) Set of all possible tokens: “Vocabulary”

Tokenization

[1] Source: https://medium.com/@abdallahashraf90x/tokenization-in-nlp-all-you-need-to-know-45c00cfa2df7
7

(+) Small vocab. size
(-) Long input sequence

## --- [Page 9] ---
•Key idea: look-up table with word embeddings

• Word embedding is defined for tokens in 𝑉:  𝑊∈ℝ|𝑉|×𝑑(𝑑: dimension of hidden feature)
• Example ( 𝑉= 4, 𝑑= 5)

Learning with Tokenized Inputs

[1] Source: https://seokhee0516.tistory.com/entry/
8

## --- [Page 10] ---
•Key idea: look-up table with word embeddings

• Word embedding is defined for tokens in 𝑉:  𝑊∈ℝ|𝑉|×𝑑(𝑑: dimension of hidden feature)

• Namely, given 𝐿input tokens 𝑥∈𝑉𝐿is converted into 𝐿word embeddings ො𝑥∈ℝ𝐿×𝑑

Learning with Tokenized Inputs

9
[1] Source: https://jalammar.github.io/illustrated-transformer/

## --- [Page 11] ---
•Key idea: look-up table with word embeddings

• Model (e.g., Transformer) will outputs ො𝑜∈ℝ𝐿×𝑑from ො𝑥
• Using 𝑊again (i.e., tied), we classify ො𝑜among tokens in vocabulary → next token (output)

Learning with Tokenized Inputs

10
[1] Source: https://jalammar.github.io/illustrated-transformer/

## --- [Page 12] ---
•Key idea: look-up table with word embeddings

• Model (e.g., Transformer) will outputs ො𝑜∈ℝ𝐿×𝑑from ො𝑥
• Using 𝑊again (i.e., tied), we classify ො𝑜among tokens in vocabulary → next token (output)

•It allows us to use same loss function (cross-entropy) to train LLMs

Learning with Tokenized Inputs

11
[1] Source: https://jalammar.github.io/illustrated-transformer/

## --- [Page 13] ---
•(After training) How to generate tokens? →K sequential inferences

12

Auto-regressive Generation

[1] Source: https://jalammar.github.io/illustrated-transformer/

## --- [Page 14] ---
•When generation is terminated and provide output?

• Using EOS token (learned during pre-training)
• E.g., [SEP] token in case of BERT

13

Auto-regressive Generation

[1] Source: https://jalammar.github.io/illustrated-transformer/

## --- [Page 15] ---
•Basics of LLM decoding

•Advanced decoding algorithms for specific goals

• Diverse beam-search
• Contrastive decoding
• Speculative decoding

Contents

14

## --- [Page 16] ---
•Basics of LLM decoding

•Advanced decoding algorithms for specific goals

• Diverse beam-search
• Contrastive decoding
• Speculative decoding

Contents

15

## --- [Page 17] ---
•Goal: For given 𝒙= [𝑥1, … , 𝑥𝐿], generating next token 𝑥𝐿+1
• From predicted output distribution Ƹ𝑝𝒙= softmax ො𝑜𝒙
= [ Ƹ𝑝1, … , Ƹ𝑝|𝑉|], σ Ƹ𝑝𝒙= 1

Generation Next Token from Distribution

[1] Source: https://jalammar.github.io/illustrated-transformer/
16

## --- [Page 18] ---
•Real-world example in pytorch:
Generation Next Token from Distribution

17
[1] Source: https://huggingface.co/blog/introducing-csearch

## --- [Page 19] ---
•Real-world example in pytorch:
Generation Next Token from Distribution

18
[1] Source: https://huggingface.co/blog/introducing-csearch

## --- [Page 20] ---
•Real-world example in pytorch:
Generation Next Token from Distribution

19

## --- [Page 21] ---
•Real-world example in pytorch:
Generation Next Token from Distribution

20

## --- [Page 22] ---
•Key idea: sequentially selecting tokens with highest probability (argmax)

• i.e., 𝑥𝐿+1 = arg max Ƹ𝑝(𝒙)
• (+) Easy-to-use, (-) it can be suboptimal depending on succeeding generations

Greedy Decoding

21
[1] Source: https://heidloff.net/article/greedy-beam-sampling/

## --- [Page 23] ---
•Key idea: handling multiple candidates and update them iteratively

• It allows us to consider multiple time steps simultaneously
• Selection criteria for update: (vanilla) likelihood 𝑝𝒙= 𝑝𝑥1 𝑝𝑥2 𝑥1 …

Beam Search

22
[1] Source: https://d2l.ai/chapter_recurrent-modern/beam-search.html

## --- [Page 24] ---
•Key idea: handling multiple candidates and update them iteratively

• It allows us to consider multiple time steps simultaneously
• Selection criteria for update: (vanilla) likelihood 𝑝𝒙= 𝑝𝑥1 𝑝𝑥2 𝑥1 …

Beam Search

23
[1] Source: https://heidloff.net/article/greedy-beam-sampling/

## --- [Page 25] ---
•Key idea: handling multiple candidates and update them iteratively

• It allows us to consider multiple time steps simultaneously
• Selection criteria for update: (vanilla) likelihood 𝑝𝒙= 𝑝𝑥1 𝑝𝑥2 𝑥1 …
• (+) More chance to generate good output, (-) significant computational costs

Beam Search

24
[1] Source: https://heidloff.net/article/greedy-beam-sampling/

## --- [Page 26] ---
•Key idea: random sampling of next token from output distribution

• i.e., 𝑥𝐿+1 = arg max Ƹ𝑝(𝒙) (greedy) →𝑥𝐿+1 ~ Ƹ𝑝(𝒙) (sampling)
• (+) More exploration during generation, (-) quality of generation can be decreased

Sampling

25
[1] Source: https://huyenchip.com/2024/01/16/sampling.html#constraint_sampling

## --- [Page 27] ---
•Key idea: inserting hyper-parameter 𝑇to adjust output distribution

• i.e., Ƹ𝑝𝒙= softmax ො𝑜𝒙/𝑇→𝑇> 1: smoothing, 𝑇< 1: sharpening

Sampling: Temperature

26
[1] Source: https://medium.com/@harshit158/softmax-temperature-5492e4007f71

## --- [Page 28] ---
•Key idea: only considering K likely tokens during sampling

• i.e., ignore Ƹ𝑝𝑖in Ƹ𝑝𝒙= [ Ƹ𝑝1, … , Ƹ𝑝|𝑉|] if it is not in top-K after sorting using probability

Sampling: Top-K

27
[1] Source: https://sooftware.io/generate/

## --- [Page 29] ---
•Key idea: only considering 𝐾likely tokens during sampling

• i.e., ignore Ƹ𝑝𝑖in Ƹ𝑝𝒙= [ Ƹ𝑝1, … , Ƹ𝑝|𝑉|] if it is not in top-K after sorting using probability

• (-) Fixed K candidates regardless of shape of output distribution

Sampling: Top-K

28
[1] Source: https://sooftware.io/generate/

## --- [Page 30] ---
•Key idea: instead of candidate number, focusing on accumulated probability

• i.e., adapt 𝐾by setting 𝐾= min𝑘(σ𝑖=1
𝑘
Ƹ𝑝𝑖

sorted 𝒙> 𝜏)

Sampling: Top-P (or Nucleus Sampling[1])

29
[1] Holtzman et al., The Curious Case of Neural Text Degeneration., ICLR 2020
[2] Source: https://sooftware.io/generate/

## --- [Page 31] ---
•Key idea: instead of candidate number, focusing on accumulated probability

• i.e., adapt 𝐾by setting 𝐾= min𝑘(σ𝑖=1
𝑘
Ƹ𝑝𝑖

sorted 𝒙> 𝜏)

Sampling: Top-P (or Nucleus Sampling[1])

30
[1] Holtzman et al., The Curious Case of Neural Text Degeneration., ICLR 2020
[2] Source: https://sooftware.io/generate/

## --- [Page 32] ---
•Example in ChatGPT
Decoding in Practice

31

## --- [Page 33] ---
•Basics of LLM decoding

•Advanced decoding algorithms for specific goals

• Diverse beam-search
• Contrastive decoding
• Speculative decoding

Contents

32

## --- [Page 34] ---
•Depending on task, goal of decoding can be changed

• E.g., debiased (de-toxic) outputs for chatbot or diverse outputs for molecule discovery

Specific Goals during Decoding

33
[1] Liu et al., DEXPERTS: Decoding-Time Controlled Text Generation with Experts and Anti-Experts., ACL 2021
[2] Jang et al., Can LLMs Generate Diverse Molecules? Towards Alignment with Structural Diversity., arXiv:24.10

## --- [Page 35] ---
•Key idea: handling multiple candidates and update them iteratively

• It allows us to consider multiple time steps simultaneously
• Selection criteria for update: (vanilla) likelihood 𝑝𝒙= 𝑝𝑥1 𝑝𝑥2 𝑥1 …

Recall: Beam Search[1]

34
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 36] ---
•Key idea: handling multiple candidates and update them iteratively

• It allows us to consider multiple time steps simultaneously
• Selection criteria for update: (vanilla) likelihood 𝑝𝒙= 𝑝𝑥1 𝑝𝑥2 𝑥1 …

Recall: Beam Search[1]

35
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 37] ---
•Goal: increasing diversity between output from different beam
Diverse Beam Search[1]

36
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 38] ---
•Key idea: augmenting beam search objective with dissimilarity term

• To this end, first dividing entire beam into multiple groups 
• Then, diversity between output of different groups is used as additional score

Diverse Beam Search[1]

37
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 39] ---
•Key idea: augmenting beam search objective with dissimilarity term

• Then, diversity between output of different groups is used as additional score

• E.g., embedding similarity or n-gram similarity

Diverse Beam Search[1]

38
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 40] ---
•Algorithm   
Diverse Beam Search[1]

39
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 41] ---
•Results: Visual Question Generation    
Diverse Beam Search[1]

40
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 42] ---
•Results: Machine Translation

• Better accuracy (BLEU-4) with much diverse outputs

Diverse Beam Search[1]

41
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 43] ---
•Qualitative results
Diverse Beam Search[1]

42
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 44] ---
•Ablation: different diversity functions

• DBS penalizes selection of tokens proportional to # of times it was selected before

Diverse Beam Search[1]

43
[1] Vijayakumar et al., Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models., AAAI 2018

## --- [Page 45] ---
•Key idea: contrasting two output distributions to obtain refined one
Contrastive Decoding[1]

44
[1] Li et al., Contrastive Decoding: Open-ended Text Generation as Optimization., ACL 2023

## --- [Page 46] ---
•Key idea: contrasting two output distributions to obtain refined one

• Here, two output distributions are from different models (e.g., larger & small)
• Assumption. Output from subtracting distribution is relatively undesirable

Contrastive Decoding[1]

45
[1] Li et al., Contrastive Decoding: Open-ended Text Generation as Optimization., ACL 2023

## --- [Page 47] ---
•But, amatuer can capture many aspects of English grammar & common sense

• Therefore, penalizing all behaviors from amateur LMs is not valid

•Idea: Adaptive Plausibility Constraint

• Namely, truncating output distribution for next token generation (similar to Top-P)
• In paper, fixed 𝛼= 0.1 is used

Contrastive Decoding[1]

46
[1] Li et al., Contrastive Decoding: Open-ended Text Generation as Optimization., ACL 2023

## --- [Page 48] ---
•Full method

• One can incorporate previous decoding methods (e.g., greedy or beam-search)

Contrastive Decoding[1]

47
[1] Li et al., Contrastive Decoding: Open-ended Text Generation as Optimization., ACL 2023

## --- [Page 49] ---
•Depending on choice of amateur, behavior of contrastive decoding is changed

• Generic performance: smaller LMs[1]

• Focusing on given input context for better RAG: same LMs but without context[2]

• Debiased or detoxic generation: toxic LMs[3]

Contrastive Decoding[1]

48

[1] Li et al., Contrastive Decoding: Open-ended Text Generation as Optimization., ACL 2023
[2] Shi et al., Trusting Your Evidence: Hallucinate Less with Context-aware Decoding., NAACL 2024 (short)
[3] Liu et al., DEXPERTS: Decoding-Time Controlled Text Generation with Experts and Anti-Experts., ACL 2021

## --- [Page 50] ---
•Experiments: continuation (input: first 32 words →output: 256 tokens)

• Metrics: (1) DIV (diversity), (2) MAUVE (similarity), (3) COH (coherence, similarity)

Contrastive Decoding[1]

49
[1] Li et al., Contrastive Decoding: Open-ended Text Generation as Optimization., ACL 2023

## --- [Page 51] ---
•Experiments: continuation (input: first 32 words →output: 256 tokens)

• Metrics: (1) DIV (diversity), (2) MAUVE (similarity), (3) COH (coherence, similarity)
• Effect of different configurations of expert and amateur models

Contrastive Decoding[1]

50
[1] Li et al., Contrastive Decoding: Open-ended Text Generation as Optimization., ACL 2023

## --- [Page 52] ---
•Experiments: continuation (input: first 32 words →output: 256 tokens)

• Human evaluation

Contrastive Decoding[1]

51
[1] Li et al., Contrastive Decoding: Open-ended Text Generation as Optimization., ACL 2023

## --- [Page 53] ---
•Experiments: summarization
Context-aware Decoding (CAD)[1]

52
[1] Shi et al., Trusting Your Evidence: Hallucinate Less with Context-aware Decoding., NAACL 2024 (short)

## --- [Page 54] ---
•This contrastive decoding framework can be generalized beyond language

• Goal: reducing hallucination of large vision-language models (LVLMs[2])

Visual Contrastive Decoding (VCD)[1]

53
[1] Leng et al., Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding., CVPR 2024
[2] Liu et al., Visual Instruction Tuning., NeurIPS 2023

## --- [Page 55] ---
•This contrastive decoding framework can be generalized beyond language

• Goal: reducing hallucination of large vision-language models (LVLMs[2])

Visual Contrastive Decoding (VCD)[1]

54
[1] Leng et al., Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding., CVPR 2024
[2] Liu et al., Visual Instruction Tuning., NeurIPS 2023

## --- [Page 56] ---
•This contrastive decoding framework can be generalized beyond language

• Goal: reducing hallucination of large vision-language models (LVLMs[2])

Visual Contrastive Decoding (VCD)[1]

55
[1] Leng et al., Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding., CVPR 2024
[2] Liu et al., Visual Instruction Tuning., NeurIPS 2023

## --- [Page 57] ---
•How to generate K tokens? →K sequential inferences of LLMs
Recap: Autoregressive Model

56
[1] Source: https://jalammar.github.io/illustrated-transformer/

## --- [Page 58] ---
•But, some tokens are easier to generate than others!!
Motivation

57
[1] Source: https://icml.cc/virtual/2023/oral/25546

Examples of hard and easy words to sequentially generate[1]

## --- [Page 59] ---
•Key idea: (1) Drafting with small model
Speculative Decoding[1]

58
[1] Leviathan et al., Fast Inference from Transformers via Speculative Decoding., ICML 2023 (Oral)
[2] Xia et al., Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding., arXiv:24.01

K token generation via auto-regressive decoding, i.e., “K inferences”

Illustration of Speculative Decoding[2]

## --- [Page 60] ---
•Key idea: (1) Drafting with small LLM (𝑀𝑞)
Speculative Decoding[1]

59
[1] Leviathan et al., Fast Inference from Transformers via Speculative Decoding., ICML 2023 (Oral)
[2] Xia et al., Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding., arXiv:24.01

K token generation via auto-regressive decoding, i.e., “K inferences”

Illustration of Speculative Decoding[2]

## --- [Page 61] ---
•Key idea: (2) Verifying drafts with large target LLM (𝑀𝑝)  
Speculative Decoding[1]

60
[1] Leviathan et al., Fast Inference from Transformers via Speculative Decoding., ICML 2023 (Oral)
[2] Xia et al., Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding., arXiv:24.01

Measuring output probabilities of draft with “only 1 inference”

Illustration of Speculative Decoding[2]

## --- [Page 62] ---
•Key idea: (2) Verifying draft with large model 
Speculative Decoding[1]

61
[1] Leviathan et al., Fast Inference from Transformers via Speculative Decoding., ICML 2023 (Oral)
[2] Xia et al., Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding., arXiv:24.01

Measuring output probabilities of draft with “only 1 inference”

Illustration of Speculative Decoding[2]

## --- [Page 63] ---
•Key idea: (2) Verifying draft with large model 
Speculative Decoding[1]

62
[1] Leviathan et al., Fast Inference from Transformers via Speculative Decoding., ICML 2023 (Oral)
[2] Xia et al., Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding., arXiv:24.01

Measuring output probabilities of draft with “only 1 inference”

Illustration of Speculative Decoding[2]

⋮

## --- [Page 64] ---
•Key idea: (2) Verifying draft with large model 
Speculative Decoding[1]

63
[1] Leviathan et al., Fast Inference from Transformers via Speculative Decoding., ICML 2023 (Oral)
[2] Xia et al., Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding., arXiv:24.01

Measuring output probabilities of draft with “only 1 inference”

Illustration of Speculative Decoding[2]

4 tokens with 1 𝑀𝑝inference (exactly same with 4 inferences) 
Theoretical guarantee

## --- [Page 65] ---
•Results: Better efficiency with small draft & simple heuristic

• (1) EnDe: English to German translation, (2) CNNDM: text summarization
• Tradeoff with larger draft model: higher acceptance rate but lower inference speed

Speculative Decoding[1]

64
Results of Speculative Decoding (Target: T5-XXL 11B), 𝛾: # of drafted tokens, 𝛼: average acceptance rate[1]

[1] Leviathan et al., Fast Inference from Transformers via Speculative Decoding., ICML 2023 (Oral)

## --- [Page 66] ---
•Qualitative examples

• Green: drafts, red: rejected draft, blue: correction by target

Speculative Decoding[1]

65
[1] Leviathan et al., Fast Inference from Transformers via Speculative Decoding., ICML 2023 (Oral)

## --- [Page 67] ---
•Guided (or controlled) decoding: consider score (reward) during decoding[1,2,3]
More Things in Decoding

66

[1] Qin et al., COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics., NeurIPS 2022
[2] Zhang et al., ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search., arXiv:24.06
[3] Mudgal et al., Controlled Decoding from Language Models., ICML 2024

## --- [Page 68] ---
Instructor: Jaehyung Kim

jaehyungk@yonsei.ac.kr

Retrieval Augmented Generation (RAG)

## --- [Page 69] ---
•Pre-training data of LLMs have certain knowledge cutoff

• E.g., 2023-10 for GPT4o (& mini)[1] and 2023-12 for LLaMA3[2]

Knowledge Cutoff with LLMs

68
[1] https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/
[2] https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md

Examples of knowledge cutoff with SOTA LLMs[1,2]

## --- [Page 70] ---
•Namely, LLMs suffer to provide answer with up-to-date knowledge
Limited Knowledge of LLMs

69

Failure case due to limitation on pre-trained knowledge (it’s asked in 2023)

## --- [Page 71] ---
•Namely, LLMs suffer to provide answer with up-to-date knowledge

• Fine-tuning could be limited to effectively incorporate new knowledge[1,2]

Limited Knowledge of LLMs

70
[1] Gekhman et al., Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?, arXiv:24.05
[2] Ren et al., Learning or Self-aligning? Rethinking Instruction Fine-tuning., arXiv:24.02

Fitting on data with new knowledge could decrease overall accuracy[1]

## --- [Page 72] ---
•Retrieving and incorporating relevant knowledge is promising way

• Namely, augmenting new knowledge as additional input of LLMs 
• Retrieve-and-read is popular way to improve QA system

Retrieval Augmentation

71

Illustration of retrieve-and-read system for ODQA[1]

[1] https://lilianweng.github.io/posts/2020-10-29-odqa/
[2] Karpukhin et al., Dense Passage Retrieval for Open-Domain Question Answering., EMNLP 2020

## --- [Page 73] ---
•Retrieving and incorporating relevant knowledge is promising way

• Namely, augmenting new knowledge as additional input of LLMs 
• Retrieval-augmented generation (RAG) now becomes standard approach

Retrieval Augmentation

72

Example of RAG with GPT-4o

## --- [Page 74] ---
•Retrieval system

•Recent approaches for improving RAG at Inference-level

• Handling long input context
• Better input query
• Robustness to noisy document

•Recent approaches for improving RAG at Training-level

Contents

73

## --- [Page 75] ---
•Retrieval system

•Recent approaches for improving RAG at Inference-level

• Handling long input context
• Better input query
• Robustness to noisy document

•Recent approaches for improving RAG at Training-level

Contents

74

## --- [Page 76] ---
•Web-search, e.g., google search[1]
Retrieval

75
[1] Page., The PageRank citation ranking: Bringing order to the web., Technical Report, 1997

## --- [Page 77] ---
•Page-rank[1]: importance is average importance of linked ones:
Retrieval: Web Search

76
[1] Source: https://sooftware.io/page_rank/

## --- [Page 78] ---
•Page-rank[1]: importance is average importance of linked ones:

• 𝑑: damping factor, i.e., 1 −𝑑is probability for jumping to random page
• Mathematically, PR(A) is probability of arriving at page A after many clicks
• Namely, you can find page-rank with following iterative method:

Retrieval: Web Search

77
[1] Page., The PageRank citation ranking: Bringing order to the web., Technical Report, 1997

## --- [Page 79] ---
•BM25[1]: raw text-level retrieval, i.e., word overlap ↑→retrieval ↑

• Given a query 𝑄, containing keywords 𝑞𝑖, BM25 score of document 𝐷is:
• 𝑁: total # of documents, 𝑛(𝑞𝑖): # of documents containing 𝑞𝑖

Text-base Retrieval

78
[1] Robertson and Zaragoza et al., The Probabilistic Relevance Framework: BM25 and Beyond., Foundations and Trends in Information Retrieval, 2009c

## --- [Page 80] ---
•Dense passage retrieval (DPR)[1]: using two sentence encoders (e.g., BERT[2])

• Similarity between query 𝑞& passage 𝑝→Dot-product of their features

• Given       
 , following contrastive loss is used:
(positive 𝑝+: labeled data,

negative 𝑝−: labeled data for difficulty + in-batch negatives)

Dense Retrieval

79
[1] Karpukhin et al., Dense Passage Retrieval for Open-Domain Question Answering., EMNLP 2020
[2] Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding., NAACL 2019

## --- [Page 81] ---
•Contriever[1]: Self-supervised training with contrastive learning

• Idea: substituting human-labeled positive as artificially generated one
• 𝑘+: positive document (crop in same document), 
𝑘𝑖: negative documents (other doc.)

Dense Retrieval

80
[1] Izacard et al., Unsupervised Dense Information Retrieval with Contrastive Learning., TMLR 2022

## --- [Page 82] ---
•There is no answer which is best retrieval method yet

• As DPR/Contriever are training base, test distribution is critical for success

Retrieval

81
[1] Kim et al., SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs., ICLR 2024

Effectiveness of different retrieval for RAG with SuRe[1]

## --- [Page 83] ---
•Then, how can we use this retrieval for RAG with LLM?

• 1. Training-base: training LLM to utilize this retrieved information to answer
• 2. Inference-base: appending retrieved information as additional input

Retrieval Augmented Generation (RAG)

82
[1] Gao et al., Retrieval-Augmented Generation for Large Language Models: A Survey., arXiv:24.05

## --- [Page 84] ---
•Then, how can we use this retrieval for RAG with LLM?

• 1. Training-base: training LLM to utilize this retrieved information to answer
• 2. Inference-base: appending retrieved information as additional input

Retrieval Augmented Generation (RAG)

83
[1] Gao et al., Retrieval-Augmented Generation for Large Language Models: A Survey., arXiv:24.05

## --- [Page 85] ---
•Retrieval system

•Recent approaches for improving RAG at Inference-level

• Handling long input context
• Better input query
• Robustness to noisy document

•Recent approaches for improving RAG at Training-level

Contents

84

## --- [Page 86] ---
•REtrieval and PLUG (REPLUG)[1]

• Assumption. Target LM (reader) is black-box, i.e., parameters can’t be accessed & updated

•Key idea. Retrieval-augmentation via simple prepending at input

• By retrieving the document with the existing method, e.g., contriever
• To further improve retrieval, new way to fine-tune with target black-box LM is proposed

Retrieval and Plug

85
[1] Shi et al., REPLUG: Retrieval-Augmented Black-Box Language Models., NAACL 2023

## --- [Page 87] ---
•Challenge. Limited context window size →Prepending all documents is difficult

•Key idea. Prepend each document separately & ensemble output probabilities

• Considering quadratic nature of Transformer, it does not increase the computation

REPLUG: Inference with Input Reformulation

86

## --- [Page 88] ---
•Challenge. Limited context window size →Prepending all documents is difficult

•Key idea. Prepend each document separately & ensemble output probabilities

• Given input 𝑥and top-K documents 𝒟′, output probability of next token 𝑦is given as:

• 𝜆(𝑥, 𝑑) is a similarity score from the used retriever model

REPLUG: Inference with Input Reformulation

87

concatenation

## --- [Page 89] ---
•Challenge. No trainable module →Limited performance

•Key idea. Update retriever to be aligned with output prob. of LM

• By using LM-Supervised signal for fine-tuning Retriever (LSR)

REPLUG: Fine-tuning Dense Retriever

88

## --- [Page 90] ---
•Key idea. Update retriever to be aligned with output prob. of LLM

1. Computing retrieval likelihood for K documents:

2. Computing LM likelihood for ground truth output 𝑦:

3. Update retriever to minimize KL divergence (i.e., consistency):

REPLUG: Fine-tuning Dense Retriever

89

## --- [Page 91] ---
•Demonstration on generic language modeling (on Pile dataset)

• Regardless of LM’s capability, REPLUG successfully improves performance (BPB)
• By fine-tuning retriever model with LSR, improvement is significantly increased

REPLUG: Experiments

90

## --- [Page 92] ---
•Results on MMLU and QA

• For both tasks, REPLUG successfully improves target LMs

REPLUG: Experiments

91

(Left) Results on MMLU and (right) QA datasets 1]

## --- [Page 93] ---
•Many retriever systems rely on training with relevance labels

• E.g., Contriever[1] also rely on labeled data for sufficient performance 
• Such dataset is often unavailable →Necessity of “zero-shot retrieval”

Motivation: Query Enhancement

92
[1] Izacard et al., Unsupervised Dense Information Retrieval with Contrastive Learning., TMLR 2022

## --- [Page 94] ---
•LLM can generate query-relevant context using leanred knowledge[1]

• Ideally, if LLM memorize all things in web, we might not need RAG
• One remaining challenge: “Hallucination”

Motivation: Query Enhancement

93
[1] Sun et al., Recitation-Augmented Language Models., ICLR 2023

## --- [Page 95] ---
•Idea: Incorporating query-relevant context from LLM

• First, generating synthetic document by prompting LLMs,
• Then, retrieving most relevant passages with existing unsupervised retriever (Contriever)

HyDE: Query Enhancement with LLMs

94
[1] Gao et al., Precise Zero-Shot Dense Retrieval without Relevance Labels., ACL 2023

Concept Figure of Hypothetical Document Embeddings (HyDE)[1]

## --- [Page 96] ---
HyDE: Query Enhancement with LLMs


| • Recap. Dense retrieval models similarity with two encoders: 𝑇 • Only passage encoder 𝐸 (∙) is used, i.e., 𝑠𝑖𝑚 𝑞, 𝑝 = 𝐸 𝑝෤ 𝑞 𝐸 (𝑝) 𝑃 𝑃 𝑃 • To improve quality, multiple documents can be generated and used: 𝑇 𝑠𝑖𝑚 𝑞, 𝑝 = ( ෍ 𝐸 𝑝෤ 𝑞 /𝐾) 𝐸 (𝑝) 𝑃 𝑘 𝑃 |  |  |
| --- | --- | --- |
|  | coder 𝐸 (∙) is used, i.e., 𝑠𝑖𝑚 𝑞, 𝑝 = 𝐸 𝑝෤ 𝑞 𝐸 𝑃 𝑃 ty, multiple documents can be generated and use 𝑇 𝑠𝑖𝑚 𝑞, 𝑝 = ( ෍ 𝐸 𝑝෤ 𝑞 /𝐾) 𝐸 (𝑝) 𝑃 𝑘 𝑃 |  |
|  | 𝑘=1,…,𝐾 | 95 |

## --- [Page 97] ---
•Contriever underperform BM25, but HyDE outperforms it

• HyDE remains competitive even when compared to fine-tuned models.

Results: Web Search

96

## --- [Page 98] ---
•HyDE again brings significant improvements to Contriever

• HyDE also shows strong performance compared to fine-tuned models

Results: Low Resource Retrieval

97

## --- [Page 99] ---
•Q. Should we use Contriever for this framework?

• A: No. BM25 is also applicable and we can improve it with better drafting

•Large language model as Retriever (LameR)[1]

• Key idea: Improving the initial generation of LLM using retriever
(Initial retrieval →Generating initial answer →Finer retrieval)

Beyond HyDE

98
[1] Chen et al., Large Language Models are Strong Zero-Shot Retriever., Findings of ACL 2024

## --- [Page 100] ---
•Large language model as Retriever (LameR)[1]

• Key idea: Improving the initial generation of LLM using retriever

Beyond HyDE

99
[1] Chen et al., Large Language Models are Strong Zero-Shot Retriever., Findings of ACL 2024

## --- [Page 101] ---
•Noisy retrieval can negatively affect LLM performance[1,2]

• E.g., RAG can be even worsen than direct answering without retrieval

Motivation: Noisy-robust RAG

100
[1] Petroni et al., How Context Affects Language Models’ Factual Predictions., AKBC 2020
[2] Li et al., Large Language Models with Controllable Working Memory., ACL 2023

## --- [Page 102] ---
•Noisy retrieval can negatively affect LLM performance[1,2]

• E.g., RAG can be even worsen than direct answering without retrieval

•Goal: Retrieval-robust LLMs

• When relevant, retrieved context should improve model performance
• When irrelevant, retrieved context should not hurt model performance.

Motivation: Noisy-robust RAG

101
[1] Petroni et al., How Context Affects Language Models’ Factual Predictions., AKBC 2020
[2] Li et al., Large Language Models with Controllable Working Memory., ACL 2023

## --- [Page 103] ---
•Key idea: Using existing NLI (Natural Language Inference) models

• Remark. SOTA model on NLI achieve strong performance (> 92%)

RetRobust: Training Free

102
[1] Yoran et al., Making Retrieval-Augmented Language Models Robust to Irrelevant Context., ICLR 2024

## --- [Page 104] ---
•Key idea: Using existing NLI (Natural Language Inference) models

• In RAG, retrieved passage →premise, question and answer →hypothesis
• Namely, correct retrieval should be entailed with question and answer
• If not entailed, direct answering without retrieval would be used instead

RetRobust: Training Free

103
[1] Yoran et al., Making Retrieval-Augmented Language Models Robust to Irrelevant Context., ICLR 2024

## --- [Page 105] ---
•Key idea: Training LLM to ignore irrelevant contexts with small data

• Previous approach could be too strict and discards relevant ones as well
• Namely, interest is collecting relevant/irrelevant passages for given query

•Q. How to construct dataset?

• Relevant passage: Top-1 passage with given retriever
• Irrelevant passages: (1) Bottom-K for given query 
or (2) Top-K for other queries

RetRobust: Small Training

104
[1] Yoran et al., Making Retrieval-Augmented Language Models Robust to Irrelevant Context., ICLR 2024

## --- [Page 106] ---
•Key idea: Training LLM to ignore irrelevant contexts with small data

• Previous approach could be too strict and discards relevant ones as well
• Namely, interest is collecting relevant/irrelevant passages for given query

•Then, model is fine-tuned to maximize likelihood of answer

• Namely, output correct prediction even under wrong retrieved passages

RetRobust: Small Training

105
[1] Yoran et al., Making Retrieval-Augmented Language Models Robust to Irrelevant Context., ICLR 2024

## --- [Page 107] ---
•Tasks: 3 different types of QA

• Single-hop QA: (1) Natural Question (NQ)
• Multi-hop QA with explicit reasoning: (2) 2WikiMQA and (3) Bamboogle
• Multi-hop QA with commonsense: (4) StrategyQA and (5) Fermi

RetRobust: Setups

106

## --- [Page 108] ---
•Tasks: 3 different types of QA

•Training: 1k for NQ and 500 for others

• Instead of full fine-tuning of LLaMA2-13B, QLoRA[1] is used
• For NLI model, BART-large fine-tuned on MNLI dataset is used

RetRobust: Setups

107
[1] Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs., NeurIPS 2023

## --- [Page 109] ---
•Tasks: 3 different types of QA

•Training: 1k for NQ and 500 for others

•Inference: Google search for retriever & 4-6 few-shot examples

RetRobust: Setups

108

## --- [Page 110] ---
•QA with Top-1 retrieval

• Incorporating NLI model successfully prevent decrease of performance
• Additional training further enlarge improvements

RetRobust: Results

109

## --- [Page 111] ---
•QA with low-rank retrieval (top) and random retrieval (bottom)

• For both scenarios, RetRobust are effective to mitigate risk of retrieval

RetRobust: Results

110

## --- [Page 112] ---
•Previously, we observe several issues in RAG

1. RAG should be adaptively applied depending on query
2. Even LLM outputs with RAG can be not grounded by retrieved documents

•Motivation. Can we directly address these issues via additional training?

• Remark. Training is most direct way to resolve the issues of deep learning

Motivation: Learning to Address Issues of RAG

111
[1] Izacard et al., Atlas: Few-shot Learning with Retrieval Augmented Language Models., TMLR 2022

## --- [Page 113] ---
•Retrieval system

•Recent approaches for improving RAG at Inference-level

• Handling long input context
• Better input query
• Robustness to noisy document

•Recent approaches for improving RAG at Training-level

Contents

112

## --- [Page 114] ---
•Self-RAG[1] uses SFT to learn LLM of adaptive RAG

• Q. Can it be improved utilizing reinforcement learning, similar to RLHF[2]?
• A. Yes (maybe also adopted by recent companies such as DeepSeek[3])

Learn Adaptive RAG via RL

113

[1] Asai et al., Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection., ICLR 2024 Oral
[2] Ouyang et al., Training Language Models to Follow Instructions with Human Feedback., NeurIPS 2022
[3] https://chat.deepseek.com/

## --- [Page 115] ---
•Search-R1[1] is open-sourced framework that apply RL to learn adaptive RAG

• As implied in the name, it mainly follows DeepSeek-R1 framework[2]

Search-R1

114
[1] Jin et al., Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning., arXiv:25.03
[2] DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning., arXiv:25.01

## --- [Page 116] ---
•Key idea of Search-R1: Incorporating RAG as output texts with special tokens

1. Reasoning tokens: <think> internal reasoning </think> 
2. Search call tokens: <search> query </search> 
3. Search results tokens: <information> retrieved documents </information>
4. Answer tokens: <answer> answer </answer>

Search-R1

115
[1] Jin et al., Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning., arXiv:25.03
[2] DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning., arXiv:25.01

## --- [Page 117] ---
•Example of answering with Search-R1
Search-R1

116

## --- [Page 118] ---
•Example of answering with Search-R1

• Interleaved CoT & retrieval is automatically emerged during RL!

Search-R1

117

## --- [Page 119] ---
•How to reward? Simple outcome reward based on ground-truth answer

• No format rewards, as it uses LLMs already demonstrates strong structural adherence

Search-R1

118

## --- [Page 120] ---
•How to reward? Simple outcome reward based on ground-truth answer

•How to train? Existing RL algorithms such as PPO and GRPO

Search-R1

119

𝑦can be from either generation by LLM or retrieval 𝑅

generation by LLM

[1] Ouyang et al., Training Language Models to Follow Instructions with Human Feedback., NeurIPS 2022
[2] DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning., arXiv:25.01


|  | retrieval 𝑅 (i.e., not generated by LLM) |
| --- | --- |


## --- [Page 121] ---
•How to reward? Simple outcome reward based on ground-truth answer

•How to train? Existing RL algorithms such as PPO and GRPO

Search-R1

120

𝑦can be from either generation by LLM or retrieval 𝑅

Removing from 
loss calculation

Only update 
for here!

[1] Ouyang et al., Training Language Models to Follow Instructions with Human Feedback., NeurIPS 2022
[2] DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning., arXiv:25.01

## --- [Page 122] ---
•Inference algorithm of Search-R1
Search-R1

121

## --- [Page 123] ---
•Setups

•Evaluation QA Datasets

Search-R1: Experiments

122

## --- [Page 124] ---
•Baselines 
Search-R1: Experiments

123

## --- [Page 125] ---
•Main results 
Search-R1: Experiments

124

## --- [Page 126] ---
•Ablation #1: PPO v.s. GRPO

•Ablation #2: Training dynamics (output length, reward under mask & Instruct)

Search-R1: Experiments

125

## --- [Page 127] ---
Instructor: Jaehyung Kim

jaehyungk@yonsei.ac.kr

Evaluation of Large Language Models

## --- [Page 128] ---
•For any system, its evaluation is important for successful deployment

•3 components in evaluation

1. Target task: what to be done by system
2. Evaluation method: which way to be used to evaluate system
3. Evaluation metric: how to measure success of system

Evaluation

127

## --- [Page 129] ---
•Example of system: delivery service

•Then, 3 components for its evaluation could be

1. Target task: deliver food to user from restaurant 
2. Evaluation method: measure delivery time 
3. Evaluation metric: average delivery time across users

Evaluation

128
Source: https://www.ytn.co.kr/_ln/0102_202404130800061445

## --- [Page 130] ---
•For deep learning system, evaluation usually relies on “test data”

• Def. Test data: data for same task and distribution, but never seen during training

•Example: fine-tuned LM (e.g., BERT[1]) for sentiment classification

Evaluation of Deep Learning Model

129
[1] Delvin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding., NAACL 2019

Illustration of BERT pre-training/fine-tuning framework[1]

“Overall, the value I got from the two hours 
watching it was the sum total of the popcorn 
and the drink. The movie was terrible.”

Negative

input
output
fine-tuned LM

## --- [Page 131] ---
•For deep learning system, evaluation usually relies on “test data”

• Def. Test data: data for same task and distribution, but never seen during training

•Example: fine-tuned LM (e.g., BERT[1]) for sentiment classification

• Target task: sentiment classification
• Evaluation method: comparing prediction on test data with human label
• Evaluation metric: average accuracy

Evaluation of Deep Learning Model

130
[1] Delvin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding., NAACL 2019

## --- [Page 132] ---
•Overview

•Evaluation of LLM with ground-truth answer

•Evaluation of LLM without ground-truth answer

Contents

131

## --- [Page 133] ---
•Overview

•Evaluation of LLM with ground-truth answer

•Evaluation of LLM without ground-truth answer

Contents

132

## --- [Page 134] ---
•LLMs are trained on massive text data of various tasks

• E.g., question answering (QA), summarization, code, etc.

•Consequently, LLM can be adapted to any task with instruction & examples

Evaluation of LLM

133
[1] Brown et al., Language Models are Few-Shot Learners., NeurIPS 2020

In-context learning (few-shot prompting) with LLMs[1]

## --- [Page 135] ---
•Two important components: (1) input prompt

• Detailed instruction + better[1,2] & more[3] examples →Better performance

Evaluation of LLM

134

[1] Zhang et al., Active Example Selection for In-Context Learning., EMNLP 2022 
[2] Rubin et al., Learning To Retrieve Prompts for In-Context Learning., NAACL 2022 
[3] Bertsch et al., In-Context Learning with Long-Context Models: An In-Depth Exploration., arXiv:24.05

In-context learning (few-shot prompting) with LLMs[1]

## --- [Page 136] ---
•Two important components: (1) input prompt

• Detailed instruction & more examples →Better performance
• Recap. Chain-of-Thought (CoT) prompting is crucial to solve complex reasoning tasks

Evaluation of LLM

135
[1] Brown et al., Language Models are Few-Shot Learners., NeurIPS 2020

In-context learning (few-shot prompting) with LLMs[1]

## --- [Page 137] ---
•Two important components: (1) input prompt & (2) decoding method

• Caution. With temperature > 0, LLM can sample different answer for same question

Evaluation of LLM

136

Temperature can be controlled by user

## --- [Page 138] ---
•Two important components: (1) input prompt & (2) decoding method

• Caution. With temperature > 0, LLM can sample different answer for same question
• Self-consistency[1]: sample multiple prediction & take majority voting

Evaluation of LLM

137
[1] Wang et al., Self-Consistency Improves Chain of Thought Reasoning in Language Models., ICLR 2023

Concept Figure of Self-consistency[1]

## --- [Page 139] ---
•LLMs are trained on massive text data of various tasks

• E.g., question answering (QA), summarization, code, etc.

•Consequently, LLM can be adapted to any task with instruction & examples
→Cons. Make difficult to evaluate LLM as we should test multiple tasks together

Evaluation of LLM

138
[1] Brown et al., Language Models are Few-Shot Learners., NeurIPS 2020

## --- [Page 140] ---
•Example: GPT-4 by OpenAI[1]
Evaluation of LLM

139
[1] OpenAI, GPT-4 Technical Report

Performance of GPT-4 on academic benchmarks[1]

## --- [Page 141] ---
•Example: GPT-4 by OpenAI[1]
Evaluation of LLM

140
[1] OpenAI, GPT-4 Technical Report

Performance of GPT-4 on academic benchmarks[1]

Target Tasks

## --- [Page 142] ---
•Example: GPT-4 by OpenAI[1]
Evaluation of LLM

141
[1] OpenAI, GPT-4 Technical Report

Performance of GPT-4 on academic benchmarks[1]

Evaluation 
Methods

## --- [Page 143] ---
•Example: GPT-4 by OpenAI[1]
Evaluation of LLM

142
[1] OpenAI, GPT-4 Technical Report

Performance of GPT-4 on academic benchmarks[1]

Evaluation 
Metric

## --- [Page 144] ---
•Example: Claude-3.5-sonnet by Antrophic[1]
Evaluation of LLM

143
[1] Source: https://www.anthropic.com/claude/sonnet

Performance of Claude-3.5-sonnet on academic benchmarks[1]

## --- [Page 145] ---
•Overview

•Evaluation of LLM with ground-truth answer

•Evaluation of LLM without ground-truth answer

Contents

144

## --- [Page 146] ---
•In open-sourced benchmark, ground truth answer is often included

• To make fair comparison be easier
• Evaluation method and metric can be different across dataset

Task with Ground Truth

145
[1] Source: https://huggingface.co/datasets

Example of benchmarks with ground truth in Huggingface (Left: MMLU, Right: Xsum)[1]

## --- [Page 147] ---
•Massive Multitask Language Understanding benchmark (MMLU[1])

• 16,000 multiple-choice questions spanning 57 academic subjects
• Over 100 million downloads as of July 2024

Task with Ground Truth: MMLU

146
[1] Hendrycks et al., Measuring Massive Multitask Language Understanding., ICLR 2021

Example of MMLU benchmark[1]

## --- [Page 148] ---
•Massive Multitask Language Understanding benchmark (MMLU[1])

• Method: few-shot prompting
• Metric: average accuracy (%)

Task with Ground Truth: MMLU

147
[1] Hendrycks et al., Measuring Massive Multitask Language Understanding., ICLR 2021

Example of Evaluation Results on MMLU benchmark[1]

## --- [Page 149] ---
•Massive Multitask Language Understanding benchmark (MMLU[1])

• Method: few-shot prompting
• Metric: average accuracy (%) →“Common metric for multiple-choice QA benchmark”

Task with Ground Truth: MMLU

148
[1] Hendrycks et al., Measuring Massive Multitask Language Understanding., ICLR 2021

Example of Evaluation Results on MMLU benchmark[1]

## --- [Page 150] ---
•Massive Multitask Language Understanding benchmark (MMLU[1])

• Caution. Output of LLM often does not follow specific format or has patterns

Task with Ground Truth: MMLU

149
[1] Hendrycks et al., Measuring Massive Multitask Language Understanding., ICLR 2021

Example of Prediction to Specific Question from MMLU benchmark[1]

Answer is in bottom →

## --- [Page 151] ---
•Massive Multitask Language Understanding benchmark (MMLU[1])

• Caution. Output of LLM often does not follow specific format or has patterns

Task with Ground Truth: MMLU

150
[1] Hendrycks et al., Measuring Massive Multitask Language Understanding., ICLR 2021

Example of Prediction to Specific Question from MMLU benchmark[1]

Answer is in top →

## --- [Page 152] ---
•Massive Multitask Language Understanding benchmark (MMLU[1])

• Caution. Output of LLM often does not follow specific format or has patterns
• Specific instruction can mitigate this (remember: not perfect!)

Task with Ground Truth: MMLU

151
[1] Hendrycks et al., Measuring Massive Multitask Language Understanding., ICLR 2021

Example of Prediction to Specific Question from MMLU benchmark[1]

←Specific 
instruction

## --- [Page 153] ---
•Natural Question dataset (NQ[1])

• 7,830 examples with 5-way annotations for development/test dataset
• LLM should generate long or short answer without multiple choices

Task with Ground Truth: NQ

152
[1] Kwiatkowski et al., Natural Questions: A Benchmark for Question Answering Research., TACL 2019

Examples in Natural Questions (NQ) benchmark[1]

## --- [Page 154] ---
•Example of (1) QA
Task with Ground Truth: NQ

153

## --- [Page 155] ---
•Example of (1) QA and (2) prediction with GPT-4o
Task with Ground Truth: NQ

154

## --- [Page 156] ---
•Example of (1) QA and (2) prediction with GPT-4o
Task with Ground Truth: NQ

155

How to evaluate 
it quantitatively
& automatically?

## --- [Page 157] ---
•Metrics for QA: Exact Match (EM) and F1 scores (Lexical Matching)

• EM: 1 if it is the exact same as its reference string, or 0 otherwise
• F1: Considering word-wise comparisons after parsing

• Precision: # words in prediction & answer / # words in prediction

• Recall: # words in prediction & answer / # words in answer

Task with Ground Truth: NQ

156

## --- [Page 158] ---
•Metrics for QA: Exact Match (EM) and F1 scores (Lexical Matching)

• EM: 1 if it is the exact same as its reference string, or 0 otherwise
• F1: Considering word-wise comparisons after parsing

• Precision: # words in prediction & answer / # words in prediction

• Recall: # words in prediction & answer / # words in answer

Task with Ground Truth: NQ

157

## --- [Page 159] ---
•Extreme summarization dataset (Xsum[1])

• 230k examples & avg. words in document/summary →430/23

Task with Ground Truth: XSum

158
[1] Narayan et al., Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization., EMNLP 2018

Example in XSum dataset1]

## --- [Page 160] ---
•Extreme summarization dataset (Xsum[1])

• 230k examples & avg. words in document/summary →430/23

Task with Ground Truth: XSum

159
[1] Narayan et al., Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization., EMNLP 2018

Example of output for document in XSum dataset1]

## --- [Page 161] ---
•Extreme summarization dataset (Xsum[1])

• 230k examples & avg. words in document/summary →430/23

Task with Ground Truth: XSum

160
[1] Narayan et al., Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization., EMNLP 2018

“Severe flooding in Scotland, particularly in Newton Stewart, Hawick, and Peeblesshire, 
has caused extensive damage to homes, businesses, and infrastructure, prompting 
calls for improved flood defenses and faster preventative measures.”

“Clean-up operations are continuing across the Scottish Borders and Dumfries and 
Galloway after flooding caused by Storm Frank.”
Ground Truth

LLM Output

## --- [Page 162] ---
•Extreme summarization dataset (Xsum[1])

• 230k examples & avg. words in document/summary →430/23

Task with Ground Truth: XSum

161
[1] Narayan et al., Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization., EMNLP 2018

“Severe flooding in Scotland, particularly in Newton Stewart, Hawick, and Peeblesshire, 
has caused extensive damage to homes, businesses, and infrastructure, prompting 
calls for improved flood defenses and faster preventative measures.”

“Clean-up operations are continuing across the Scottish Borders and Dumfries and 
Galloway after flooding caused by Storm Frank.”
Ground Truth

LLM Output

How to measure similarity 
between these sentences?

## --- [Page 163] ---
•ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

• Measures overlap of words or phrases between generated and reference texts
• E.g., Rouge-N: measuring overlap of N-grams

Task with Ground Truth: XSum

162
[1] Narayan et al., Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization., EMNLP 2018

## --- [Page 164] ---
•Distance (L2 or cosine) at embedding space

• First, raw texts are mapped to embedding using separately trained sentence encoder[1]

• Then, one can evaluate semantical similarity by measuring distance
• Caution. Quality of sentence embedding is very important (check [2])

Task with Ground Truth: XSum

163
[1] Reimers and Gurevych., Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks., EMNLP 2019
[2] https://huggingface.co/spaces/mteb/leaderboard

Illustration of Sentence-BERT to Calculate Cosine Similarity between Two Sentences1]

## --- [Page 165] ---
•Overview

•Evaluation of LLM with ground-truth answer

•Evaluation of LLM without ground-truth answer

Contents

164

## --- [Page 166] ---
•Many real-world tasks often do not have fixed ground truth

• Example: summarization

Task with No Ground Truth

165
[1] Source: https://www.anthropic.com/claude/sonnet

“Severe flooding in Scotland, particularly in Newton Stewart, Hawick, and Peeblesshire, 
has caused extensive damage to homes, businesses, and infrastructure, prompting 
calls for improved flood defenses and faster preventative measures.”

“Clean-up operations are continuing across the Scottish Borders and Dumfries and 
Galloway after flooding caused by Storm Frank.”
Ground Truth

LLM Output

## --- [Page 167] ---
•Many real-world tasks often do not have fixed ground truth

• Example: summarization or story generation[1] (continuation)

Task with No Ground Truth

166
[1] Holtzman et al., The Curious Case of Neural Text Degeneration., ICLR 2020

2008년 이후 서호주 해안에서 전례 없이 많은 수의 어린 고
래가 좌초되었습니다.

## --- [Page 168] ---
•Many real-world tasks often do not have fixed ground truth

• Example: summarization or story generation (continuation)

Task with No Ground Truth

167
[1] Holtzman et al., The Curious Case of Neural Text Degeneration., ICLR 2020

워싱턴 주에서 운영되는 포경 그물에 전례 없이 많은 수의 
새끼 고래가 잡혔습니다.

서호주 해안에서 영양 부족으로 인해 좌초된 혹등고래의 
수가 증가하고 있다고 수의학 연구자들이 밝혔습니다.

2008년 이후 서호주 해안에서 전례 없이 많은 수의 어린 고
래가 좌초되었습니다.

## --- [Page 169] ---
•Def. Tasks which create coherent text that continues from given context

• LLM’s output can be viewed as open-ended text generation

Open-ended Text Generation

168
[1] Holtzman et al., The Curious Case of Neural Text Degeneration., ICLR 2020

Example Generations Continuing Initial Sentence under Different Decoding Methods[1]

## --- [Page 170] ---
•Def. Tasks which create coherent text that continues from given context

• LLM’s output can be viewed as open-ended text generation

Open-ended Text Generation

169
[1] Holtzman et al., The Curious Case of Neural Text Degeneration., ICLR 2020

Example Generations Continuing Initial Sentence under Different Decoding Methods[1]

How can we evaluate?

## --- [Page 171] ---
•Def. Tasks which create coherent text that continues from given context

• LLM’s output can be viewed as open-ended text generation

Open-ended Text Generation

170
[1] Holtzman et al., The Curious Case of Neural Text Degeneration., ICLR 2020

Evaluation Results of Different Decoding Methods[1]

## --- [Page 172] ---
•Perplexity measures how well model can predict sequence of words

• Probabilities are (1) from generating LLM itself or (2) external evaluating LLM 
• Lower perplexity usually indicates better predictive performance (quality)
• Caution. It may not always correlate with human-perceived quality

Metrics for Open-ended Text Generation

171
[1] Holtzman et al., The Curious Case of Neural Text Degeneration., ICLR 2020

## --- [Page 173] ---
•But, what if one has certain evaluation criteria?

• E.g., clarity, coherence, and creativity
• Not just calculate separately, but want to incorporate into single score
• Hiring human annotator is most accurate, but too costly

Metrics for Open-ended Text Generation

172

## --- [Page 174] ---
•G-Eval [1]: using LLMs to assess the quality of outputs of LLM

• User first gives (1) task information and (2) evaluation criteria

Metrics for Open-ended Text Generation

173
[1] Liu et al., G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment., EMNLP 2023

## --- [Page 175] ---
•G-Eval [1]: using LLMs to assess the quality of outputs of LLM

• User first gives (1) task information and (2) evaluation criteria
• With these inputs, LLM generates evaluation steps automatically

Metrics for Open-ended Text Generation

174
[1] Liu et al., G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment., EMNLP 2023

## --- [Page 176] ---
•G-Eval [1]: using LLMs to assess the quality of outputs of LLM

• User first gives (1) task information and (2) evaluation criteria
• With these inputs, LLM generates evaluation steps automatically
• Then, LLM can evaluate arbitrary input according to evaluation criteria and task info.

Metrics for Open-ended Text Generation

175
[1] Liu et al., G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment., EMNLP 2023

## --- [Page 177] ---
•G-Eval with GPT-4 exhibits highest correlation to human evaluation

• Compare to previous metrics like ROUGE or BERTScore (embedding)

Metrics for Open-ended Text Generation

176
[1] Liu et al., G-EVAL: NLG Evaluation using GPT-4 with Better Human Alignment., EMNLP 2023

Correlation of Model-base Evaluation and  Human Evaluation[1]

## --- [Page 178] ---
•G-Eval with GPT-4 exhibits highest correlation to human evaluation

• Compare to previous metrics like ROUGE or BERTScore (embedding)
• Using LLM as evaluator with specific criteria becomes standard approach[1]

Metrics for Open-ended Text Generation

177
[1] Yuan et al., Self-Rewarding Language Models., ICML 2024
Prompt to use LLM as Scoring Model[1]

## --- [Page 179] ---
•Relative evaluation is another way to find good single output

• Idea: pairwise (or more) comparison among candidate and pick the best one

Relative Evaluation

178
[1] Ouyang et al., Training Language Models to Follow Instructions with Human Feedback., NeurIPS 2022

Pairwise Comparison by Human Annotator to Decide Good Output by LLM[1]

LLM’s generation

## --- [Page 180] ---
•Example: Chatbot Arena[1]

• Considered as one of most reliable comparison between LLMs using real-user feedback

Relative Evaluation

179
[1] https://lmarena.ai/

Interface of Chatbot Arena for Relative Evaluation between LLMs[1]

## --- [Page 181] ---
•Example: Chatbot Arena[1]

• Considered as one of most reliable comparison between LLMs using real-user feedback

Relative Evaluation

180
[1] https://lmarena.ai/

Interface of Chatbot Arena for Relative Evaluation between LLMs[1]

## --- [Page 182] ---
•Example: Chatbot Arena[1]

• Considered as one of most reliable comparison between LLMs using real-user feedback

Relative Evaluation

181
[1] https://lmarena.ai/

Interface of Chatbot Arena for Relative Evaluation between LLMs[1]

## --- [Page 183] ---
•Example: Chatbot Arena[1]

• Considered as one of most reliable comparison between LLMs using real-user feedback

Relative Evaluation

182
[1] https://lmarena.ai/

Leaderboard of Chatbot Arena[1]

## --- [Page 184] ---
•Pairwise comparison using human annotator is common way to evaluate

• Example: Llama (LLM from Meta)[1]

Relative Evaluation

183
[1] https://lmarena.ai/

Comparison between Llama2 and other LLMs[1]

## --- [Page 185] ---
•Pairwise comparison using human annotator is common way to evaluate

• Example: Llama (LLM from Meta)[1]

• Cons. Cost for large human annotation

Relative Evaluation

184
[1] https://lmarena.ai/

Comparison between Llama2 and other LLMs[1]

## --- [Page 186] ---
•Solution: LLM-as-judge[1]

• Idea: replacing human annotator by prompting LLM

LLM-as-Judge

185
[1] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena., NeurIPS 2023 (Dataset and Benchmark Track)

Example of prompt for pairwise comparison by LLM[1]

## --- [Page 187] ---
•Solution: LLM-as-judge[1]

• Key advantages:
(1) Scalability and 
(2) Explainability

LLM-as-Judge

186
[1] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena., NeurIPS 2023 (Dataset and Benchmark Track)

Example of pairwise comparison via GPT-4[1]

## --- [Page 188] ---
•Solution: LLM-as-judge[1]

• Key advantages: High agreement between GPT-4 and humans
• MT-Bench: 80 high-quality multi-turn questions from 8 categories, e.g., roleplay, math

LLM-as-Judge

187
[1] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena., NeurIPS 2023 (Dataset and Benchmark Track)

Example of question in MT-Bench Roleplay[1]

Turn #1: "Pretend yourself to be Elon Musk in all the following conversations. 
Speak like Elon Musk as much as possible. Why do we need to go to Mars?",

Turn #2: "How do you like dancing? Can you teach me?"

## --- [Page 189] ---
•Solution: LLM-as-judge[1]

• Key advantages: High agreement between GPT-4 and humans
• E.g., GPT-4 & Human: 85% (among human: 81%)

LLM-as-Judge

188
[1] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena., NeurIPS 2023 (Dataset and Benchmark Track)

Agreement between two types of judges on MT-Bench (S2: no tie)[1]

## --- [Page 190] ---
•Caution. LLM judge exhibits several biases

1. Position bias: favor certain positions over others, e.g., 1st presentation > 2nd one

LLM-as-Judge

189
[1] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena., NeurIPS 2023 (Dataset and Benchmark Track)

Example of position bias[1]

## --- [Page 191] ---
•Caution. LLM judge exhibits several biases

1. Position bias: favor certain positions over others, e.g., 1st presentation > 2nd one
2. Verbosity bias: favors longer responses, even if they are not as clear or high-quality

LLM-as-Judge

190
[1] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena., NeurIPS 2023 (Dataset and Benchmark Track)

Example of verbosity bias under “repetitive list” attack[1]

## --- [Page 192] ---
•Caution. LLM judge exhibits several biases

1. Position bias: favor certain positions over others, e.g., 1st presentation > 2nd one
2. Verbosity bias: favors longer responses, even if they are not as clear or high-quality

LLM-as-Judge

191
[1] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena., NeurIPS 2023 (Dataset and Benchmark Track)

Example of verbosity bias under “repetitive list” attack[1]

## --- [Page 193] ---
•Caution. LLM judge exhibits several biases

1. Position bias: favor certain positions over others, e.g., 1st presentation > 2nd one
2. Verbosity bias: favors longer responses, even if they are not as clear or high-quality
3. Self-preference bias: favor answers generated by themselves

LLM-as-Judge

192
[1] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena., NeurIPS 2023 (Dataset and Benchmark Track)

Experiments that reveal self-preference bias[1]

## --- [Page 194] ---
•Solution for position bias: two evaluations with swapped positions[1]

• Then, take average as result, e.g., 1st: (A), 2nd: (B) →“Tied”
• (-) 2x more costs

LLM-as-Judge

193
[1] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena., NeurIPS 2023 (Dataset and Benchmark Track)

## --- [Page 195] ---
•Solution for position bias: two evaluations with swapped positions[1]

•Solution for verbosity bias: removing effect of length[1]

• AlpacaEval[1]: using GPT-4 as judge to evaluate alignment
• LC Win Rate: “length-controlled” win rate

LLM-as-Judge

194
[1] Dubois et al., Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators., arXiv:24.04

## --- [Page 196] ---
•Solution for position bias: two evaluations with swapped positions[1]

•Solution for verbosity bias: removing effect of length[1]

• AlpacaEval[1]: using GPT-4 as judge to evaluate alignment
• LC Win Rate: “length-controlled” win rate

LLM-as-Judge

195
[1] Dubois et al., Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators., arXiv:24.04

High Correlation between LC AlpacaEval (by GPT-4) & Chatbot Arena (by human)[1]