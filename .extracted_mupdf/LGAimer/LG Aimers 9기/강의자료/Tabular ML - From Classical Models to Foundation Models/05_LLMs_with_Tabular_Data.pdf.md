## --- [Page 1] ---
#5. LLMs with Tabular Data

Instructor: Hankook Lee @ Efficient Learning Lab., Sungkyunkwan University

1

## --- [Page 2] ---
Lecture Overview

• This lecture covers the use LLMs with tabular data
• What LLMs bring to Tabular ML?

• Tables as Language: Serialization and Prediction
• LIFT (2022), TableLLM (2023), TABLET (2023)

• LLMs as Semantic Components in Tabular ML
• CAAFE (2023), FeatLLM (2024), OCTree(2024), DeLTa (2025)

• LLMs for Tabular Anomaly Detection
• AnoLLM (2025), ReTabAD (2026), AutoAnoEval (2026)

• LLM-based tabular foundation models
• TABULA (2024)

2

## --- [Page 3] ---
Large Language Models (LLMs)

• LLMs are trained to understand and generate text at scale
• Learn semantic relationships among words, concepts, and descriptions
• Follow natural-language instructions for new tasks
• Adapt to new problems from zero or a few demonstrations
• Provide useful prior knowledge about real-world entities and concepts

• However, tables are not naturally written as text
• A tabular row must be serialized in a language-compatible form
• Numerical values and structured relationships may remain difficult for LLMs

Question:
Can the semantic capabilities of LLMs be useful for tabular data?

3

## --- [Page 4] ---
What LLMs bring to Tabular ML?

• In previous lectures, most tabular models mainly learn from observed 
values and statistical patterns

• LLMs can additionally leverage textual semantics and prior knowledge
• Semantic understanding of columns and values
• E.g., "BMI" and "Diagnosis Code" carry interpretable meanings
• Common and domain-related knowledge
• Cities, occupations, diseases, products, and institutions may provide useful context
• Natural-language task specification
• The label meanings and decision context can be stated directly in a prompt
• Zero-shot and few-shot adaptation
• Predictions may be made with little or no task-specific training data
• Semantic reasoning for anomaly detection
• Abnormality may depend on contextual meaning, not only unusual numerical values
4

## --- [Page 5] ---
What LLMs bring to Tabular ML?

• In previous lectures, most tabular models mainly learn from observed 
values and statistical patterns

• LLMs can additionally leverage textual semantics and prior knowledge
• Semantic understanding of columns and values
• E.g., "BMI" and "Diagnosis Code" carry interpretable meanings
• Common and domain-related knowledge
• Cities, occupations, diseases, products, and institutions may provide useful context
• Natural-language task specification
• The label meanings and decision context can be stated directly in a prompt
• Zero-shot and few-shot adaptation
• Predictions may be made with little or no task-specific training data
• Semantic reasoning for anomaly detection
• Abnormality may depend on contextual meaning, not only unusual numerical values
5

LLMs extend tabular ML 
from learning numerical patterns 
to reasoning over semantic context

## --- [Page 6] ---
LLM Workflow

6
Fang et al., Large Language Models (LLMs) on Tabular Data: Prediction, Generation, and Understanding – A Survey, TMLR 2024

## --- [Page 7] ---
Main Questions for This Lecture

• Can an LLM directly predict from tabular data?
• Serialize rows as text and use the LLM as a predictor

• Can an LLM improve another tabular predictor?
• Use semantic knowledge for feature engineering or hybrid pipelines

• Can an LLM help detect semantically abnormal samples?
• Use likelihood scoring, restored context, or semantic evaluation

• Can we build an LLM-based tabular foundation model?
• Pre-train on many tables and transfer to unseen tasks

7

## --- [Page 8] ---
Main Questions for This Lecture

• Can an LLM directly predict from tabular data?
• Serialize rows as text and use the LLM as a predictor

• Can an LLM improve another tabular predictor?
• Use semantic knowledge for feature engineering or hybrid pipelines

• Can an LLM help detect semantically abnormal samples?
• Use likelihood scoring, restored context, or semantic evaluation

• Can we build an LLM-based tabular foundation model?
• Pre-train on many tables and transfer to unseen tasks

8

## --- [Page 9] ---
Tables Must Become Text

• LLMs do not directly process tables as structured inputs

• Before prediction, a table must be turned into a token sequence
• This step is called serialization (or linearization)

• The serialization format is part of the model design
• It affects what structure the LLM can see and how many tokens are spent

9

## --- [Page 10] ---
Common Serialization Methods

10
Fang et al., Large Language Models (LLMs) on Tabular Data: Prediction, Generation, and Understanding – A Survey, TMLR 2024

## --- [Page 11] ---
LIFT: Language Interface for Tabular Tasks

• Key Idea: Convert non-language ML tasks into language-input and 
language-output tasks

• No architecture or loss modification
• Show that LLMs can be fine-tuned through a natural-language interface

11
Dinh et al., LIFT: Language-Interfaced Fine-Tuning for Non-language Machine Learning Tasks, NeurIPS 2022

## --- [Page 12] ---
TabLLM: Tabular Classification with LLMs

• Key Idea: Serialize a tabular row into a natural-language string with 
a short task description

• Use LLMs for zero-shot or few-shot classification

12
Hegselmann et al., TabLLM: Few-shot Classification of Tabular Data with Large Language Models, AISTATS 2023

## --- [Page 13] ---
TabLLM: Tabular Classification with LLMs

• Key Idea: Serialize a tabular row into a natural-language string with 
a short task description

• Show that input serialization matters

13
Hegselmann et al., TabLLM: Few-shot Classification of Tabular Data with Large Language Models, AISTATS 2023

## --- [Page 14] ---
TABLET: Learning from Instructions

• Key Idea: Evaluate how natural-language instructions help LLMs 
solve tabular prediction tasks

• Show that instructions can improve zero/few-shot tabular prediction

14
Slack & Singh, TABLET: Learning From Instructions For Tabular Data, 2023

## --- [Page 15] ---
TABLET: Learning from Instructions

• Key Idea: Evaluate how natural-language instructions help LLMs 
solve tabular prediction tasks

• Show that instructions can improve zero-shot tabular prediction
• Generated instructions can be also useful

15
Slack & Singh, TABLET: Learning From Instructions For Tabular Data, 2023

## --- [Page 16] ---
Summary: Serialization & Prediction

• Main strategy: Convert tabular prediction into language prediction
• Strength: uses semantic prior knowledge and task descriptions
• Weakness: inference cost, context limits, serialization sensitivity

• When using LLMs, input design matters
• Row serialization
• Task description
• Natural-language instructions

• LLMs can directly solve tabular prediction problems, 
but this is not always practical

• LLMs may struggle with numerical values, long tables, and inference cost

16

## --- [Page 17] ---
Main Questions for This Lecture

• Can an LLM directly predict from tabular data?
• Serialize rows as text and use the LLM as a predictor

• Can an LLM improve another tabular predictor?
• Use semantic knowledge for feature engineering or hybrid pipelines

• Can an LLM help detect semantically abnormal samples?
• Use likelihood scoring, restored context, or semantic evaluation

• Can we build an LLM-based tabular foundation model?
• Pre-train on many tables and transfer to unseen tasks

17

## --- [Page 18] ---
Beyond Direct LLM Prediction

• Direct LLM prediction is often impractical
• Serialize every row & query LLM for each prediction
• Sensitive to prompt length and input format
• Expensive for large-scale inference

• Alternative: Use LLMs as semantic components
• Generate useful features
• Improve feature-generation rules
• Refine decision-tree rules

Let standard ML models perform final prediction

instead of asking the LLM to predict every row

18

## --- [Page 19] ---
Semantic Feature Engineering with LLMs

• Why feature engineering?
• Tabular performance often depends on meaningful feature transformations
• LLMs can use column names, value meanings, and task descriptions
• This allows feature generation beyond fixed search spaces

• Common patterns
• LLM proposes feature transformations
• A standard ML model evaluates them
• Useful features are kept for final prediction

LLMs can act as feature engineers that translate semantic

knowledge into usable columns

19

## --- [Page 20] ---
CAAFE: Feature Engineering via LLM

• Key Idea: Use an LLM to generate semantically meaningful features
• Provide dataset context, column names, and task description
• LLM generates Python code for new features

20
Hollmann et al., Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering, NeurIPS 2023

## --- [Page 21] ---
CAAFE: Feature Engineering via LLM

• Key Idea: Use an LLM to generate semantically meaningful features
• Provide dataset context, column names, and task description
• LLM generates Python code for new features

21
Hollmann et al., Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering, NeurIPS 2023

## --- [Page 22] ---
CAAFE: Feature Engineering via LLM

• Key Idea: Use an LLM to generate semantically meaningful features
• Generated features can improve performance in downstream tasks
• Better LLMs can generate better features

22
Hollmann et al., Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering, NeurIPS 2023

## --- [Page 23] ---
LLM as Few-Shot Feature Engineer

• Key Idea: Use LLMs to generate predictive features from few-shot 
examples

• LLM proposes rule-based features → train a simple predictor
• Different LLM-generated rules capture different semantic hypotheses
→ Ensemble multiple predictors

23
Han et al., Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning, ICML 2024

## --- [Page 24] ---
LLM as Few-Shot Feature Engineer

• Key Idea: Use LLMs to generate predictive features from few-shot 
examples

24
Han et al., Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning, ICML 2024

## --- [Page 25] ---
Feature Generation with Decision Tree

• Key Idea: Use LLMs as optimizers for feature-generation rules with 
decision tree reasoning

25
Nam et al., Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning, NeurIPS 2024

## --- [Page 26] ---
Feature Generation with Decision Tree

• Key Idea: Use LLMs as optimizers for feature-generation rules with 
decision tree reasoning

26
Nam et al., Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning, NeurIPS 2024

with language descriptions

## --- [Page 27] ---
Feature Generation with Decision Tree

• Key Idea: Use LLMs as optimizers for feature-generation rules with 
decision tree reasoning

27
Nam et al., Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning, NeurIPS 2024

without language descriptions

## --- [Page 28] ---
Feature Generation with Decision Tree

• Key Idea: Use LLMs as optimizers for feature-generation rules with 
decision tree reasoning

• LLMs with decision tree reasoning successfully generate meaningful features 
with and without language descriptions

• Why decision tree reasoning helps?
• Decision trees provide compact feedback to the LLM
• Validation score alone is weak feedback
• Tree rules provide interpretable reasoning about previous feature attempts
• LLM can refine future feature rules using this feedback

28
Nam et al., Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning, NeurIPS 2024

## --- [Page 29] ---
DeLTa: LLM Meets Decision Trees

• Key Idea: Use an LLM to generate a rule for enhancing decision tree
• Ask the LLM to generate a refined rule from multiple decision-trees
• Use the rule to calibrate the prediction of original decision trees

29
Ye et al., LLM Meeting Decision Trees on Tabular Data, NeurIPS 2025

## --- [Page 30] ---
DeLTa: LLM Meets Decision Trees

• Key Idea: Use an LLM to generate a rule for enhancing decision tree
• Ask the LLM to generate a refined rule from multiple decision-trees
• Use the rule to calibrate the prediction of original decision trees

• Relying on only decision tree rules, not semantic priors of column names

30
Ye et al., LLM Meeting Decision Trees on Tabular Data, NeurIPS 2025

## --- [Page 31] ---
Summary: LLMs as Semantic Components

• Main strategy: Use LLMs to improve tabular models rather than 
directly predict every sample

• Generate semantic features
• Optimize feature-generation and decision rules

• Practical benefits:
• Avoids serializing and querying the LLM for every test row
• Reduces inference cost and latency
• Retains the efficiency of standard tabular models

• LLMs can inject semantic knowledge into efficient tabular pipelines 
without serving as the final predictor

31

## --- [Page 32] ---
Main Questions for This Lecture

• Can an LLM directly predict from tabular data?
• Serialize rows as text and use the LLM as a predictor

• Can an LLM improve another tabular predictor?
• Use semantic knowledge for feature engineering or hybrid pipelines

• Can an LLM help detect semantically abnormal samples?
• Use likelihood scoring, restored context, or semantic evaluation

• Can we build an LLM-based tabular foundation model?
• Pre-train on many tables and transfer to unseen tasks

32

## --- [Page 33] ---
LLMs for Tabular Anomaly Detection

• Anomalies are often domain-specific and context-dependent
• Rare values are not always anomalous
• Common values can be suspicious in the wrong context

• Traditional AD models mostly rely on raw feature values
• They may miss semantic inconsistencies
• They often ignore column descriptions, units, and domain knowledge

Key Question:
Can LLMs help define what is abnormal using semantic context?

33

## --- [Page 34] ---
LLM Likelihood for Anomaly Scoring

• Key Idea: Use a language model to model normal tabular rows
• Convert tabular rows into standardized text
• Fine-tune a pre-trained LLM on normal data
• Low likelihood → abnormal data

34
Tsai et al., AnoLLM: Large Language Models for Tabular Anomaly Detection, ICLR 2025

## --- [Page 35] ---
LLM Likelihood for Anomaly Scoring

• Key Idea: Use a language model to model normal tabular rows
• Convert tabular rows into standardized text
• Fine-tune a pre-trained LLM on normal data
• Low likelihood → abnormal data

• AnoLLM can handle mixed-type tabular data

• Limitations
• Requires fine-tuning LLM on normal data
• Inference can be expensive
• Likelihood may not always align with semantic abnormality

35
Tsai et al., AnoLLM: Large Language Models for Tabular Anomaly Detection, ICLR 2025

## --- [Page 36] ---
ReTabAD: Restoring Semantic Context

• Key Idea: Evaluate how metadata helps tabular anomaly detection
• Problem: Existing tabular AD benchmarks mainly provide raw feature values
• They lack textual metadata such as feature descriptions and domain knowledge

36
Yoon et al., ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection, ICLR 2026

## --- [Page 37] ---
ReTabAD: Restoring Semantic Context

• Key Idea: Evaluate how metadata helps tabular anomaly detection
• Problem: Existing tabular AD benchmarks mainly provide raw feature values
• They lack textual metadata such as feature descriptions and domain knowledge
• This paper builds a benchmark with semantically enriched tabular datasets

37
Yoon et al., ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection, ICLR 2026

## --- [Page 38] ---
ReTabAD: Restoring Semantic Context

• Key Idea: Evaluate how metadata helps tabular anomaly detection
• Problem: Existing tabular AD benchmarks mainly provide raw feature values
• They lack textual metadata such as feature descriptions and domain knowledge
• This paper builds a benchmark with semantically enriched tabular datasets

38
Yoon et al., ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection, ICLR 2026

## --- [Page 39] ---
ReTabAD: Restoring Semantic Context

• Key Idea: Evaluate how metadata helps tabular anomaly detection
• Semantic metadata enables context-aware "zero-shot" anomaly detection

39
Yoon et al., ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection, ICLR 2026

## --- [Page 40] ---
ReTabAD: Restoring Semantic Context

• Key Idea: Evaluate how metadata helps tabular anomaly detection
• Semantic metadata enables context-aware "zero-shot" anomaly detection
• Also enables more interpretable reasoning

40
Yoon et al., ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection, ICLR 2026

## --- [Page 41] ---
AutoAnoEval: Model Selection w/o Labels

• AD model selection is difficult when anomaly labels are unavailable
• Reliable evaluation requires both normal and anomalous samples
• True anomalies may be absent even in validation sets
• Different detectors may perform well on different types of anomalies

• Question: Can we generate abnormal samples for model evaluation?
• Generated anomalies should be semantically meaningful
• They should also be diverse, ranging from obvious outliers to subtle 
anomalies near the normal boundary

41
Yoon et al., AutoAnoEval: Semantic-Aware Model Selection via Tree-Guided LLM Reasoning for Tabular Anomaly Detection, EACL Findings 2026

## --- [Page 42] ---
AutoAnoEval: Model Selection w/o Labels

• Key Idea: Generate pseudo-anomaly via tree-guided LLM reasoning
• Normal decision paths can be extracted by decision trees
• LLM refines anomaly conditions by modifying the normal decision paths

42
Yoon et al., AutoAnoEval: Semantic-Aware Model Selection via Tree-Guided LLM Reasoning for Tabular Anomaly Detection, EACL Findings 2026

## --- [Page 43] ---
AutoAnoEval: Model Selection w/o Labels

• Key Idea: Generate pseudo-anomaly via tree-guided LLM reasoning
• Normal decision paths can be extracted by decision trees
• LLM refines anomaly conditions by modifying the normal decision paths

• The generated evaluation set offers reasonable performance estimates 
that closely approximate those on real-world anomalies

43
Yoon et al., AutoAnoEval: Semantic-Aware Model Selection via Tree-Guided LLM Reasoning for Tabular Anomaly Detection, EACL Findings 2026

## --- [Page 44] ---
Summary: Tabular Anomaly Detection

• LLMs can support tabular AD in different ways:
• Anomaly scoring: Model normal rows as text and detect unlikely samples
• Context-awareness: Use semantic metadata to reason about abnormality
• Model selection: Generate meaningful pseudo-anomalies for evaluation

• Why are LLMs useful?
• Anomalies are often defined by domain context, not only by statistical rarity
• LLMs can use feature descriptions and domain knowledge

LLMs extend tabular anomaly detection 
from value-based scoring to semantic reasoning and evaluation.

44

## --- [Page 45] ---
Main Questions for This Lecture

• Can an LLM directly predict from tabular data?
• Serialize rows as text and use the LLM as a predictor

• Can an LLM improve another tabular predictor?
• Use semantic knowledge for feature engineering or hybrid pipelines

• Can an LLM help detect semantically abnormal samples?
• Use likelihood scoring, restored context, or semantic evaluation

• Can we build an LLM-based tabular foundation model?
• Pre-train on many tables and transfer to unseen tasks

45

## --- [Page 46] ---
Toward Tabular Foundation Models

• Can we build a tabular foundation model that can transfer across 
many tables and tasks?

• Why is this difficult?
• Tables are highly heterogeneous
• Unlike text, tabular tasks are not naturally standardized
• Large-scale training requires unifying many tables into a common format

• Two directions:
• LLM-based tabular foundation models
• TabPFN-style foundation models (next lecture)

46

## --- [Page 47] ---
TabuLa-8B: LLM-based Tabular FM

• Key Idea: Fine-tune a large language model on a massive corpus of 
serialized tabular prediction tasks

• Base model: Llama 3-8B
• Training dataset (T4): 2.1B rows from over 4M unique tables
• Evaluation: 329 unseen datasets

47
Gardner et al., Large Scale Transfer Learning for Tabular Data via Language Modeling, NeurIPS 2024

## --- [Page 48] ---
Summary

• LLMs can exploit semantic information in tables, but direct 
prediction is not always practical

• LLMs enable broader applications in tabular ML
• Semantic feature engineering and rule refinement
• Context-aware anomaly detection
• Foundation models transferable across tables and tasks

• The key is using semantic knowledge while preserving the 
efficiency and reliability of tabular modeling

48

## --- [Page 49] ---
Thank You for Your Attention!

49