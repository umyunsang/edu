## --- [Page 1] ---
Lightweight LLM beyond Scaling:
From Business Cost Optimization to Frontier Research

Youngjae Yu

SNU PI Lab

youngjaeyu@snu.ac.kr

## --- [Page 2] ---
Session 1 : A Business-Centric Approach 
to LLM Cost Reduction and Efficiency

•
Understanding LLM lightweighting and 
productization trends among US Big Tech 
and startups

•
Quantifying business-oriented efficiency 
metrics (cost, latency, throughput) for 
informed decision-making

•
Sharing architecture-level trade-offs and 
lightweighting techniques across LLM 
designs

## --- [Page 3] ---
https://www.hankyung.com/article/2025102134571

## --- [Page 4] ---
Global Small Language Model (SLM) Market Size

[1]https://market.us/report/large-language-model-powered-tools-market
[2]https://www.databridgemarketresearch.com/reports/global-small-
language-model-slm-market

## --- [Page 5] ---
Global Lightweight Language Model Ecosystem

https://www.marketsandmarkets.com/Market-Reports/small-
language-model-market-4008452.html

## --- [Page 6] ---
AI companies are copying each other’s homework to 
make cheap models

•
AI companies copy LLMs with  ‘distillation’

•
The price of building AI is falling to new lows.

[1]https://www.ft.com/content/c117e853-d2a6-4e7c-aea9-e88c7226c31f

[2]https://www.businessinsider.com/deepseek-openai-distillation-big-tech-trouble-
cheap-commodity-ai-2025-3

## --- [Page 7] ---
Cost and resource constraints 
are new battlefield

•
As LLM adoption becomes 
widespread, cost and resource 
constraints are becoming major 
bottlenecks for enterprises.

•
Companies are aggressively slashing 
LLM prices, triggering a supply-side 
cost war in the AI model market.

## --- [Page 8] ---
LMSYS Chatbot Arena

https://lmarena.ai/leaderboard

## --- [Page 9] ---
https://objectbox.io/the-rise-of-small-language-models/

## --- [Page 10] ---
Adopting Small & Open LLMs

•
Open-source LLMs are emerging as a cost-efficient option through 
business-specific deployment models, including on-premises 
installations and private-cloud hosting.

•
Model compression, serving, optimization 
techniques are reducing inference costs, 
hardware requirements, and energy 
consumption.

Exploring the Dynamic Landscape of Large Language Models: An Insightful 
Overview by Prashanth Ramshetti

## --- [Page 11] ---
What Makes Small Language Models So Attractive?

Accessible and Affordable

They can be run (in inference mode) on limited resource regimes

Easier to Customize

Small models can typically be fine-tuned on just a single GPU.

More Energy Efficient

Small language models require fewer computational resources, making them more energy-
efficient. It also means reduced energy consumption

Cheaper to Develop

These models only require a relatively small number of GPUs.

Valuable for Educational Purposes

They are more manageable and thus easier to understand and tweak.

## --- [Page 12] ---
What Makes Small Language Models So Attractive?

•
Lightweight language models enable significantly lower training and 
inference costs compared to large-scale models.

•
On smartphones, robots, cars, and wearables, latency is the user 
experience. large models often feel slow because of server and network 
bottlenecks.

•
Users perceive smartness more from fast, responsive behavior than 
from sheer model size.

## --- [Page 13] ---
On-device on Mainstream

•
Lightweight models can be 
deployed locally, 
on-device, or in 
environments with limited 
connectivity

•
This enables privacy-
sensitive use cases (e.g., 
healthcare, enterprise 
internal systems)

https://drlee.io/the-on-device-ai-revolution-why-your-next-mobile-app-needs-a-
small-multimodal-language-model-876fad370c1c

## --- [Page 14] ---
On-device on Mainstream

•
Talkback -  Gemini Nano’s multimodal capabilities to improve image 
descriptions for blind and low vision users.

•
Pixel Recorder - Gemini Nano with Multimodality model enables 
support for longer recordings and higher quality summaries.

## --- [Page 15] ---
On-device on Mainstream

Big tech is betting on small, efficient models that live on phones and PCs, 
not just in giant data centers.

•
Apple: 2025 tech report introduces a 3B on-device model with KV-cache 
sharing and 2-bit QAT, plus a PT-MoE server model on Private Cloud 
Compute.

•
Microsoft: the Mu model, a 330M encoder–decoder optimized for NPUs 
on Copilot+ PCs  runs Windows Settings entirely on-device.

•
Chrome: Gemini Nano inference extends to CPUs, so built-in AI works 
on many more laptops with the same Prompt API.

## --- [Page 16] ---
Headlines That Set the Stage

•
Providers cut your bill for reused context.

○
Prompt Caching discounts cached tokens.

•
Global race for efficient AI

○
low-cost, high-quality models spark a price/perf shake-up.

○
DeepSeek had deliberately "distilled" OpenAI's models into its own.

•
More efficient "mini" and reasoning-optimized models for daily use.

○
price/performance pressure that rewards lightweight deployments.

Gim et al. Prompt Cache: Modular Attention Reuse for 
Low-Latency Inference

## --- [Page 17] ---
Headlines That Set the Stage

•
Alibaba Qwen3: open-weight family with hybrid thinking modes (/think 
vs /no_think) and MoE options, designed for efficient real-world 
deployment on modest hardware

○
Efficiency breakthroughs are coming from outside the US too

○
They all lean on smaller, smarter, often MoE-based models.

https://qwen.ai/blog?id=qwen3-livetranslate

## --- [Page 18] ---
The Cheapest Tokens Are Cached or Local

•
OpenAI: prompt caching now gives 90% cheaper cached input tokens, 
with 24-hour retention for GPT-5.1 and friends.

○
OpenAI realtime-mini: text models with 10× cheaper cached tokens

○
(e.g., $0.06/1M cached tokens) and lower per-million costs than full-size 
realtime models.

•
Anthropic: Claude Haiku 4.5 prices start at $1 / $5 per million (in/out) 
with up to 90% savings from caching and 50% from batch APIs.

○
Which positioned as the “fastest, most cost-efficient” Claude

➢Now, providers are literally paying you to reuse context and batch

work instead of brute-forcing with bigger models.

## --- [Page 19] ---
Small Models by Design

Start with small-by-default; go bigger only when SLOs require.

•
Microsoft Phi-3: a family of small language models aimed 
at cost-effective quality.

○
“Outperform same-size and next-size up” across common tasks.

•
Apple Intelligence uses small on-device models for privacy 
& responsiveness.

○
Private Cloud Compute handles heavier tasks with strict privacy 
guarantees.

○
push simple, frequent tasks on-device; surge to cloud 
when needed

## --- [Page 20] ---
Cloud vs on-device?

It’s small-on-device + efficient cloud, with smart routing and strict SLOs.

•
Apple ships custom AI servers for Private Cloud Compute, blending local 
NPUs with privacy-preserving cloud offload.

•
These servers use Apple’s own foundation models plus a paid Gemini 
backend for some Siri tasks, picking the right model for the right job and 
budget.

https://www.tomshardware.com/desktops/servers/apples-houston-
built-ai-servers-now-shipping

## --- [Page 21] ---
Pixel Recorder moved On-Device

•
The Recorder app on Pixel sees a 24% boost in engagement with 
Gemini Nano-powered feature

○
Faster, private summaries

○
small models unlock new UX and reduce per-request cost.

https://android-developers.googleblog.com/2024/08/recorder-app-on-pixel-
sees-boost-in-engagement-with-gemini-nano.html

## --- [Page 22] ---
Serving Breakthrough, vLLM & PagedAttention

•
PagedAttention reduces KV-cache waste and enables continuous 
batching.

•
Academic & industrial benchmarks show large throughput gains at 
similar latency.

•
Business impact: same GPUs, more QPS makes lower $/answer.

https://arxiv.org/abs/2309.06180

## --- [Page 23] ---
Decoding Gets Faster :  Speculative/Drafting

•
Draft-and-verify methods can reduce decoding steps by ~2× in practice.

•
Pair with vLLM for multiplicative gains. We’ll revisit this later session

https://research.google/blog/looking-
back-at-speculative-decoding/

## --- [Page 24] ---
Putting It All Together: 
Why Lightweight LLMs Matter for Us

•
Economics: token prices, caching discounts, and small-model SKUs 
mean your unit economics can improve by 2~5 x without inventing 
new models.

•
Tech trend: from Apple, Microsoft, Google, DeepSeek, and Alibaba, the 
common pattern is ‘small-by-default, big-when-needed.’

## --- [Page 25] ---
How Small Models + Big Models Actually Collaborate

•
SLM handles 60–90% of easy queries

•
LLM handles only high-complexity reasoning

•
Router system: confidence-based escalation

•
Example: intent >> extraction >> toolcalling >> reasoning

https://www.themoonlight.io/ko/review/smaller-smarter-closer-the-edge-of-collaborative-generative-ai

## --- [Page 26] ---
Small Language Models are the Future of Agentic AI

•
Modern AI agents increasingly perform small, repetitive, specialized 
tasks rather than open-ended conversation.

•
Despite this, most agents still rely on a single large general-purpose 
LLM for all sub-tasks.

•
Small Language Models (SLMs) are sufficiently powerful, more 
operationally suitable, and significantly more economical

•
LLMs should be used sparingly for only the tasks requiring broad 
general reasoning.

Belcak et al. Small Language Models are the Future of Agentic AI

## --- [Page 27] ---
SLMs Are Powerful Enough

Phi-3 Small (7B)
– Matches or surpasses 70B models in language understanding & code generation.
Nemotron-H (2–9B)
– Instruction-following and coding accuracy comparable to 30B dense LLMs.
SmolLM2 (125M–1.7B)
– Reaches performance of 14B contemporaries; rivals 70B models from two years prior.
DeepSeek-R1-Distill (1.5–8B)
– 7B version outperforms Claude 3.5 Sonnet and GPT-4o-0513 in reasoning.
RETRO-7.5B
– GPT-3-level performance using retrieval despite being 25x smaller.
Hymba-1.5B
– Outperforms some 13B models in instruction accuracy; 3.5x faster throughput.

Model size is no longer the limiting factor. 
>> Architecture, training data, and 
inference-time reasoning matter

## --- [Page 28] ---
How SLMs + LLMs Work Together

LM Agency (Left)
A single model (LLM or SLM) orchestrates 
planning and tool calling.
Suitable when a general reasoning root node is 
needed.

Code Agency (Right)
A controller program orchestrates the flow.
Multiple specialized SLMs handle subtasks 
(formatting, extraction, tool calling).
Only complex reasoning escalates to an LLM

➢SLMs for 60~90% of calls
➢LLMs only for high-

complexity reasoning
➢Lower cost, reduced

hallucinations, higher 
predictability.

## --- [Page 29] ---
SLMs align with how real agent systems actually work

. Cost & Efficiency

•
Running a 7B model is 10–30× cheaper than a 70–175B model (latency, FLOPs, energy).

•
Minimal or zero GPU parallelism required. lower infra cost.

•
Fine-tuning SLMs takes hours, not weeks.

. Flexibility & Modularity

•
Agents naturally break tasks into small subtasks.

•
SLMs can be specialized per task (intent parsing, extraction, tool calling, code generation, etc.).

•
Faster updates to support new formats, regulations, or behaviors.

. Narrow Functional Needs

•
Most agent interactions use only very small slices of LM ability

## --- [Page 30] ---
Agent Workflows Naturally Generate 
Training Data for SLMs

•
Agent interactions produce structured signals: prompts, tool calls, 
responses, and success or failure traces.

•
These signals form high-quality, task-specific datasets with minimal manual 
labeling.

•
Logs allow clustering of recurring behaviors and identification of SLM-
specializable subtasks.

•
Fine-tuned SLMs become increasingly accurate as the agent accumulates more 
workflow data.

•
This creates a self-improving loop where SLMs gradually replace LLM calls.

## --- [Page 31] ---
Efficiency is the Strategy

•
Users feel latency, CFOs see token & GPU bills.

•
Lightweight models + smart serving hit both pain points at once.

## --- [Page 32] ---
Efficiency First: The Business Case to Reduce LLM Cost

•
Most LLM costs scale with tokens and idle GPU time.

•
Small, fast models can meet many SLO (Service Level Objective)s at a 
fraction of cost when paired with smart serving.

•
Cut tokens, choose cheaper models, cache & batch, and instrument cost 
KPIs.

vs

## --- [Page 33] ---
Where the Money Goes

•
API spend: Input/Output tokens × model rates (+ caching discounts).

•
Self-host: GPU-hours × $/GPU-hr + power + engineering + utilization 
effects.

•
Hidden factors: context length, decoding strategy, batching efficiency, 
cache hit rate.

## --- [Page 34] ---
Model Pricing Snapshot

For example,

•
OpenAI GPT-4.1: Input $3.00 / 1M tokens, Output $12.00 / 1M tokens.

•
OpenAI GPT-4.1 mini: Input $0.80 / 1M tokens, Output $3.20 / 1M tokens.

•
Anthropic Claude Sonnet (4.x): Input $3.00 / 1M, Output $15.00 / 1M.

•
Assume 500 input + 250 output tokens per request.

○
GPT-4.1 ≈ $0.0045 per request , ~$4,500/month at 1M req.

○
GPT-4.1 mini ≈ $0.0012 per request, ~$1,200/month at 1M req.

○
Switching to mini in this workload saves ≈ 73.3%.

## --- [Page 35] ---
Prompt Caching & Token Hygiene

•
Provider caching can discount repeated input tokens and reduce 
latency.

•
Design for reuse: shared system prompts, retrieval templates, 
instruction libraries.

•
Token hygiene:

○
shorten context

○
dedupe docs

○
constrain output with JSON schemas.

## --- [Page 36] ---
Choose Smaller Models Early (Distill Later)

•
Start with smaller models that satisfy baseline quality.

•
Use distillation to close the quality gap if needed

•
Keep escape hatch: route hard queries to larger models.

https://www.geeksforgeeks.org/machine-learning/knowledge-distillation/

## --- [Page 37] ---
Serving Engine Matters (Throughput & Money
)

•
vLLM with PagedAttention reduces KV cache waste & boosts batching.

○
Throughput can be multiples higher vs naive HF serving (same hardware).

•
Features to look for:

○
Continuous batching, prefix/prompt caching, chunked prefill.

•
Drafting predicts several future tokens per step.

○
Speculative decoding verifies drafts with the base model to cut steps.

•
Expect speedups in decode-bound workloads; tune for quality vs speed.

## --- [Page 38] ---
Self-Host Reference Points (GPU-hr)

•
GCP T4 lists around $0.35/GPU-hr

○
newer GPUs (L4, A100/H100) cost more but run faster.

•
Batching + vLLM/TensorRT-LLM can make self-hosting 
cost-competitive at scale.

•
Utilization is the key

○
idle GPUs erase the advantage.

## --- [Page 39] ---
Cost Simulation: Traffic Scales

•
500 input + 250 output tokens per request
•
GPT-4.1 vs GPT-4.1 mini vs L4 GPU self-host

Case 1 . API: GPT-4.1 vs GPT-4.1 mini

Case 2. Self-host (L4 / vLLM)

○
GPU-hour: $1.2/hr (L4 on GCP)

○
At 80% utilization, cost ~ $5.8K/month for handling ~5M requests

○
At 50% utilization, cost jumps to $9.3K/month

Monthly Requests
GPT-4.1 API Cost
GPT-4.1 mini API Cost
Savings

50,000
~$225
~$60
73%↓

500,000
~$2,250
~$600
73%↓

5,000,000
~$22,500
~$6,000
73%↓

## --- [Page 40] ---
Cost Simulation: Traffic Scales

•
500 input + 250 output tokens per request
•
GPT-4.1 vs GPT-4.1 mini vs L4 GPU self-host

Case 1 . API: GPT-4.1 vs GPT-4.1 mini

Case 3 . Hybrid Strategy
•
API for spikes (10–20% of traffic)
•
Self-host for steady baseline (80–90%)
>> Balanced cost: $6~7K/month
>> Best latency + predictable cost + elasticity

Monthly Requests
GPT-4.1 API Cost
GPT-4.1 mini API Cost
Savings

50,000
~$225
~$60
73%↓

500,000
~$2,250
~$600
73%↓

5,000,000
~$22,500
~$6,000
73%↓

## --- [Page 41] ---
Make or Buy?

•
LOW volume / HIGH variability -> API first (pay per token).

•
HIGH volume / STEADY traffic -> consider self-host (optimize $/GPU-hr).

•
Hybrid: API for spikes & specialty models; self-host for steady workloads.

## --- [Page 42] ---
Top Cost Levers (Checklist)

•
Model choice: small-by-default; route hard cases to large.

•
Token control: shorter prompts, better RAG chunking, JSON outputs.

•
Caching: provider prompt caching; prefix & KV cache reuse.

•
Serving: vLLM/TensorRT-LLM; continuous batching; pinned batch sizes.

•
Quantization (4/8-bit) & lightweight fine-tuning (LoRA/QLoRA).

•
Speculative/drafting decoding; early stop on confidence.

•
Traffic shaping: micro-batch windows; SLA-aware queuing.

## --- [Page 43] ---
Design Patterns We Can Follow..

Small-by-default, route-up when needed

○
Default to o-mini / Haiku / small custom models

○
route rare hard cases to a larger model

On-device for frequent, low-risk tasks

○
Mu for Windows Settings, Apple 3B for UI tasks, Gemini Nano for browser helpers.

Exploit caching & KV optimizations

○
Design prompts to maximize cache reuse; consider KV compression/pruning techniques.

Use distillation & fine-tuning to close quality gaps

○
Start from a small open or proprietary model and distill from a stronger teacher.

## --- [Page 44] ---
Summary

Why Lightweight LLMs Became Essential

•
Scaling laws are bending: small, efficient models now match or exceed 
last-generation LLMs.

•
Lightweight models improve cost, latency, energy efficiency, and 
privacy by default.

•
Big tech shift: “small-on-device + efficient-cloud” with smart routing 
and strict SLOs.

## --- [Page 45] ---
Summary

The New Efficiency Stack: Models × Serving × Tokens

•
Efficiency = (Model choice) × (Serving engine) × (Token strategy).

•
vLLM + PagedAttention = 2–8× throughput gains at same latency.

•
Speculative/MEDUSA decoding = 2–3× decoding efficiency.

•
Prompt caching & token hygiene reduce cost at scale.

## --- [Page 46] ---
Summary

What this mean for us? Our business approach?

•
Start small-by-default; distill or route up only when SLOs demand.

•
Adopt on-device + efficient-cloud hybrid deployment for privacy & 
scale.

•
Instrument cost KPIs and iterate

○
caching, batching, quantization, distillation.

•
Lightweight LLMs are not just cheaper

○
They unlock new UX and new product classes.

## --- [Page 47] ---
Session 2 : U.S. Research Trends in Lightweight LLM

How Global big techs are making models 
cheaper, smaller, and more efficient

•
Focus on recent (2025) research & 
product launches from U.S. companies

•
See how OpenAI, Anthropic, Microsoft, 
Apple, Google push efficiency

•
Extract design patterns we can reuse in 
our own lightweight LLM

## --- [Page 48] ---
Bird eye view : Efficiency Megatrends

•
Small-by-default models: o3-mini & o4-mini (OpenAI), Claude Haiku 
4.5 (Anthropic), Mu (Microsoft), Apple’s 3B on-device model.

•
On-device & hybrid: Apple Intelligence, Windows on-device agent, 
Gemini Nano on Android/Chrome with Prompt API and CPU inference.

•
Infra & caching: 90%-off prompt/context caching (OpenAI GPT-5.1, 
Claude), KV-cache optimization and MoE server models.

○
Architectures evolved for fast inference: Princeton’s hardware-efficient 
attention (GLA/GTA), new decoding kernels, and KV-cache research matured.

## --- [Page 49] ---
Open AI          : Cost-efficient reasoning model

•
o3-mini (Jan 2025): strong STEM/coding with low cost & latency

•
o4-mini (Apr 2025): smaller model optimized for fast, cost-efficient 
reasoning, top performance on AIME 2024/2025 benchmarks.

•
GPT-5.1 prompt caching (2025): cached input tokens 90% cheaper (e.g. 
$0.125/M vs $1.25/M), extended 24h retention. Pattern: small reasoning 
models + caching fundamentally change $/query economics.

OpenAI frames these as reasoning-oriented small models, studied under its 
preparedness and safety frameworks.

## --- [Page 50] ---
Anthropic

•
Claude Haiku 4.5 (Oct 2025): Anthropic’s fastest, most cost-efficient 
model, matching or surpassing Sonnet 4

https://www.anthropic.com/news/claude-haiku-4-5

## --- [Page 51] ---
Anthropic          : Lightweight LLMs & Multi-agent orchestration

•
Claude Haiku 4.5 (Oct 2025): Anthropic’s fastest, most cost-efficient 
model, matching or surpassing Sonnet 4

•
Cost is about 1/3 the price of Sonnet 4 and 1/15 of Opus, targeted at 
high-volume enterprise workloads.

•
Multi-agent orchestration : Sonnet 4.5 can break a task into subtasks 
and coordinate a “team of Haikus”

https://skywork.ai/blog/agentic-coding-claude-haiku-4-5-
beginners-guide-sub-agent-orchestration/

## --- [Page 52] ---
Microsoft        :  Mu & On-Device NPU Research

•
Mu language model (Jun 2025): 330M encoder–decoder LM optimized 
for NPUs on Copilot+ PCs, enabling the Windows Settings agent to run 
fully on-device.

•
Training recipe: hybrid dataset (~3.6M examples) + noise injection, 
instruction tuning, and LoRA for robustness & efficiency.

•
Broader research: on-device NPU inference systems  (NPU offloading to 
cut prefill latency) influence this design space.

https://blogs.windows.com/windowsexperience/2025/06/23/introducing-
mu-language-model-and-how-it-enabled-the-agent-in-windows-settings

## --- [Page 53] ---
By pairing state-of-the-art quantization techniques with 
hardware-specific optimizations

Encoder-Decoder Architecture (Mu) compared 
to Decoder-only Architecture (GPT)
https://blogs.windows.com/windowsexperience/2025/06/23/introducing-
mu-language-model-and-how-it-enabled-the-agent-in-windows-settings

“It has the fast token throughputs and ultra-fast time to 
first token responses despite the large amount of input 
context provided to the model.”

## --- [Page 54] ---
Apple        : On-device, efficiency and privacy

•
Apple Intelligence Foundation Language Models: Tech Report 2025 
introduces two core models

○
An ∼3B-parameter on-device multilingual, multimodal model with KV-cache 
sharing and 2-bit quantization-aware training (QAT), optimized for Apple 
silicon

○
A scalable Parallel-Track Mixture-of-Experts (PT-MoE) server model with 
interleaved global-local attention for Apple’s Private Cloud Compute.

•
Research contributions: architectural tweaks, data curation, and 
inference optimizations explicitly framed around efficiency and privacy.

## --- [Page 55] ---
Google        : Gemini Nano as a Local AI Runtime

•
Android AICore & Gemini Nano: system-level service running on-device 
LLMs for features like summaries, replies, and translations, optimized 
for low-latency.

•
Chrome Prompt API (2025): lets web apps call Gemini Nano directly in 
the browser; recent updates add CPU inference so more laptops can run 
local AI.

•
Research direction: treat Nano as a standard local inference layer, while 
heavier Gemini models live in the cloud for complex tasks.

## --- [Page 56] ---
https://menlovc.com/perspective/2025-mid-year-llm-market-update/

## --- [Page 57] ---
https://menlovc.com/perspective/2025-mid-year-llm-market-update/

## --- [Page 58] ---
https://menlovc.com/perspective/2025-mid-year-llm-market-update/

## --- [Page 59] ---
https://menlovc.com/perspective/2025-mid-year-llm-market-update/

## --- [Page 60] ---
https://menlovc.com/perspective/2025-mid-year-llm-market-update/

## --- [Page 61] ---
https://menlovc.com/perspective/2025-mid-year-llm-market-update/

## --- [Page 62] ---
Serving Breakthrough                   : vLLM V1

•
Zero-overhead prefix caching in V1 (near-zero CPU cost even at 0% 
hit-rate)

•
On by default in V1 engine; simple migration from V0

•
TPU backend: unified JAX + PyTorch path, broader coverage

•
Lesson: engine-first wins (TPS/QPS gains without model changes)

## --- [Page 63] ---
Academic Focus : KV Cache is the Bottleneck

KV cache = the model’s short-term memory of past tokens

It avoids re-computing attention for the whole history

But it grows with context → big memory & latency cost

KV memory dominates long-context latency/cost

Tackle with quantization/compression + smart policies

## --- [Page 64] ---
Academic Focus : Algorithms for Faster Decode

•
A small draft model guesses several tokens

•
The main model quickly verifies or fixes them

•
Fewer total steps -> faster decoding

➢ReDrafter (ICLR’25): state-of-the-art speculative drafting

➢Diffusion-based drafting (NAACL’25): parallelize draft + verify

➢Hierarchical/training-free variants emerge

➢Engine hooks: vLLM / TensorRT-LLM speculative decoding

## --- [Page 65] ---
Academic Focus : Hardware-Efficient Attention

•
GQA (Grouped-Query Attention): many Q heads, but fewer K/V heads 
shared -> smaller KV cache

•
KV memory shrinks roughly in proportion to K/V heads

•
Use when context is long or VRAM is tight

## --- [Page 66] ---
Academic Focus : Hardware-Efficient Attention

•
GTA: matches GQA quality with ~1/2 KV footprint

•
GLA: parallel-friendly latent attention; up to ~2× decoding-kernel 
speedup vs FlashMLA

•
Use when context is long & KV dominates

Overview of Grouped-Tied Attention (GTA)

## --- [Page 67] ---
KVzip: Compressing KV Without Retraining

•
3~4× smaller KV around 2× faster decoding

•
Query-agnostic: one compressed KV works across prompts

•
Drop-in for long-chat products

https://www.arxiv.org/abs/2505.23416

## --- [Page 68] ---
Decoding Gets Faster :  Speculative/Drafting

•
Draft-and-verify methods can reduce decoding steps by ~2× in practice.

•
Great for decode-bound workloads, with minor quality tuning.

https://medium.com/@genai.works/speed-up-llm-inference-with-
speculative-decoding-1fc79701e9d6

## --- [Page 69] ---
Decoding Gets Faster :  Speculative/Drafting

https://research.google/blog/looking-back-at-speculative-decoding/

## --- [Page 70] ---
There are keywords along with SLMs

Most potential for scaling LLMs

Agent
 Test-Time Scaling
 Synthetic Data
 Post-training
 Inference-time reduction

## --- [Page 71] ---
Agent! we're at the start of the decade of agents

https://www.businessinsider.com/andrej-karpathy-ai-agents-timelines-openai-2025-10

## --- [Page 72] ---

## --- [Page 73] ---
Test-Time Compute

## --- [Page 74] ---
https://arcprize.org/blog/oai-o3-pub-breakthrough

## --- [Page 75] ---
https://arcprize.org/blog/oai-o3-pub-breakthrough

## --- [Page 76] ---
Post-Training

Luo et al. A Survey on Efficient Large Language Model Training: From Data-centric  Perspectives

## --- [Page 77] ---
Post-Training

https://github.com/luo-junyu/Awesome-Data-Efficient-LLM

## --- [Page 78] ---
Synthetic Data :Llama 3.1

## --- [Page 79] ---
Synthetic Data

## --- [Page 80] ---
Session 3 : Breaking Scaling Law – Distillation from 
Large-Scale Intelligence to Lightweight Deployment

•
Scaling Law

•
Why distillation now

•
Supervised KD (Forward KL)

•
Synthetic-data distillation

•
On-policy / Generalized KD

•
Multimodal Distillation

•
Takeaways

## --- [Page 81] ---
Arms race on LLM Size vs Performance

https://labelyourdata.com/articles/llm-fine-tuning/llm-model-size

## --- [Page 82] ---
•
Increasing parameter count improves reasoning, comprehension, and generalization

•
Models between 7B and 13B parameters deliver a strong balance of speed, accuracy, 
and cost efficiency

Model Size Range
Typical Tasks
Performance
Trade-Offs

1–3B
Simple NLP, 
embeddings, mobile 
inference

Fast, limited 
reasoning

Shallow context 
understanding

7–13B
General chat, 
summarization, QA
Strong balance
Moderate compute 
cost

30–70B
Advanced reasoning, 
multilingual, code 
generation

High accuracy
Requires enterprise 
GPUs

100B+
Multimodal, research-
scale models
Peak performance
Very high cost and 
latency

Arms race on LLM Size vs Performance

## --- [Page 83] ---
Background of Arms race

Scaling Laws for Language model

Scaling Laws for Autoregressive Generative Modeling

Why the scaling law matters?

## --- [Page 84] ---
Background of big multimodal AI

88

Scaling Laws for Language model

Scaling Laws for Autoregressive Generative Modeling

Why the scaling law matters?

What matters, what doesn’t matter for ML performance
What problems should we work on?
What should we expect in the future?

Maybe, we can make progress with building a better engine 
rather than improving the main hypothesis

## --- [Page 85] ---
Scaling Laws for Language Models

89

There are precise scaling laws for performance of ML 
models
As a function of

Parallel pathways Model parameters, N
Dataset size, D
Total compute used for training, C

Kaplan et al. Scaling Laws for Neural Language Models

## --- [Page 86] ---
Scaling Laws for Language Models

90

Power-law relationships with each individual factor when not 
bottlenecked by the other two.
In each case, all of the other quantities are much larger.
Ex) for model size N scaling, we must have very large dataset D

Kaplan et al. Scaling Laws for Neural Language Models

## --- [Page 87] ---
Achieve scaling, avoid bottleneck

91

Performance mostly about avoiding bottlenecks

Not enough data
Not enough parameter
Not enough compute
Bad information propagation through the network design
(Resnet, Transformer, Batchnorm solve this bottleneck)

If we already have a good scalable architecture (i.e. transformer)
and optimization method, many other details don’t matter much.

just change of constant prefactors

## --- [Page 88] ---
New Motivation – Other modalities

92

Previously,
We observed scale-law relationships from LMs [1]

Do they apply to all data modalities?

Image, Video, Math …

How do improvements on the loss translate to

Improvements in representation quality
Performance on downstream tasks?

From there, What else can we learn from them?

[1] Kaplan et al. Scaling Laws for Neural Language Models

## --- [Page 89] ---
Multimodal

93

Transformer

Transformer
applies to all data modalities

## --- [Page 90] ---
Smooth scaling of reducible loss across domains

94

Transformer with an autoregressive cross-entropy loss
Scaling law apply to generative modeling across a wide variety of data modalities

Irreducible loss L ∞ is a fitted domain-dependent const

## --- [Page 91] ---
Information theoretic interpretation

95

model sizes N, compute budgets C, or dataset sizes D,
The scaling relation for the loss
𝐿(𝑥)=𝐿_∞+(𝑥_0/𝑥)^(𝛼_𝑥) ≈S(True)+ D_KL (True||Model)

𝑥=𝑁, 𝐶, 𝐷

𝛼_𝑥is a modality-dependent exponent

𝐿_∞ is irreducible loss,
estimates the entropy of the true data distribution

•
𝐱𝟎

𝐱

𝛂𝐱

Reducible loss estimate of the KL divergence between the true and

model distributions

## --- [Page 92] ---
Information theoretic interpretation

96

•
The scaling relation for the loss

L x = L∞+ x0

x

αx

≈S True + DKL(True||Model)

## --- [Page 93] ---
Then what is the optimal model size?

97

•
The optimal model size for a given compute budget.
•
“Opt Model Size vs Compute” relation is very nearly a pure power-law

𝑁_𝑜𝑝𝑡 (𝐶)∝𝐶^𝛽,  𝛽 ~ 0.7

## --- [Page 94] ---
Information theoretic interpretation

98

## --- [Page 95] ---
Finetuning lmage GMs to ImageNet Classification

99

•
The smooth trends for finetuned performance on image classification

downstream performance also improves with model size and compute

## --- [Page 96] ---
Finetuning lmage GMs to ImageNet Classification

100

Model parameters, N
Dataset size, D
Total compute for training, C

체급이깡패다

## --- [Page 97] ---
Upcoming trends : Architecture is less important

104

Except when the architecture itself creates bottleneck.
Past : Solve a bad information propagation problem, (i.e. architecture, solver)
Next : Now let’s scale up

Kaplan et al. Scaling Laws for Neural 
Language Models

## --- [Page 98] ---
Have we hit a wall?

## --- [Page 99] ---
Ilya Sutskever at NeurIPS 2024

## --- [Page 100] ---

## --- [Page 101] ---
Clean Data : The fossil fuel of AI?

## --- [Page 102] ---
https://epoch.ai/trends#data

## --- [Page 103] ---
pretraining scaling laws are not exactly breaking down, but perhaps slowing down
pre-training is just one part of the puzzle

## --- [Page 104] ---
Can we bend the Scaling Law?

https://x.com/swyx/status/1830866865884991999/photo/1

## --- [Page 105] ---
The Crack in 'Bigger is Always Better'

•
Cost–performance frontier shows smaller/cheaper models catching up.

•
Distillation helps shift the frontier by transferring knowledge to 
compact students.

https://x.com/swyx/status/1830866865884991999/photo/1

## --- [Page 106] ---
Distillation Scaling Laws

•
Predict how well a distilled student LM 
will perform, given a fixed compute 
budget.

•
A “distillation scaling law” that 
predicts student cross-entropy from 
student size, distillation tokens, and 
teacher quality.

•
I’ll revisit this paper later!

Busbridge et al. Distillation Scaling Laws

## --- [Page 107] ---
Why Distillation Now

•
Smaller 'mini' models are often deployed for speed and cost with 
modest quality gaps.

•
Knowledge Distillation (KD) enables students to approach teacher 
quality.

•
Our goal: VRAM down, speed up, minimal quality drop.

https://www.geeksforgeeks.org/machine-learning/knowledge-distillation/

## --- [Page 108] ---
Distillation in a nutshell

https://www.linkedin.com/posts/richard-chanlongfai_it-is-not-a-meme-about-the-
distillation-technique-activity-7291476808787447809-yvw5
Li et al. Symbolic Chain-of-Thought Distillation: Small Models Can Also 
"Think" Step-by-Step

## --- [Page 109] ---
Knowledge Distillation

•
Transfer the probability distribution (knowledge) from Teacher to Student.

•
Match the next-token distribution (soft labels), typically via Forward KL.

•
Soft targets convey confidence structure beyond hard labels.

## --- [Page 110] ---
Methods at a Glance

•
A. Supervised KD (Forward KL)

•
B. Synthetic-Data Distillation (Sequence-level SFT on teacher outputs)

•
C. On-Policy / Generalized KD (Reverse KL; student generates, teacher 
scores)

The Magic of LLM Distillation - Rishabh Agarwal, Google DeepMind

## --- [Page 111] ---
Supervised Distillation

## --- [Page 112] ---
Supervised Distillation for Language Models

The Magic of LLM Distillation - Rishabh Agarwal, Google DeepMind

## --- [Page 113] ---
(A) Supervised KD: Mechanics & Trade-offs

•
Mechanics: minimize KL(P_teacher ‖ P_student) over logits with 
temperature.

○
Pros: fine-grained token-level alignment; strong when tokenizers match.

○
Cons: logit tensors are large; long sequences can be memory-heavy; usually requires 
same tokenizer.

## --- [Page 114] ---
(B) Synthetic-Data KD: Practical Workhorse

•
Pipeline: prompts >> Teacher answers >> SFT Student on (prompt, 
answer) pairs.

•
Only needs API access; tokenizers may differ; avoids logit storage.

•
Case: Gemma 2B distilled from 7B beats from-scratch and lowers 
perplexity.

## --- [Page 115] ---
Make Synthetic Data Better (BoN & Cost-Matching)

•
Best-of-N (BoN): aka Rejection Sampling. sample N teacher answers; 
keep only the best to train the student.

•
Compute/Cost-matched: use a cheaper teacher but increase N; 
competitive at lower cost.

•
Higher-quality teacher outputs leads better students.

1. Sample N model generations,
2. Score them,
3. Keep the one with highest score

Bansal et al. Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-
Optimal Sampling. ICLR 2025.

Wang et al. Don't Throw Away Data: Better Sequence Knowledge Distillation. 
ICLR 2025

## --- [Page 116] ---
Make Synthetic Data Better (BoN & Cost-Matching)

•
Best-of-N (BoN): aka Rejection Sampling.

•
Compute/Cost-matched

## --- [Page 117] ---
(C) On-Policy / Generalized KD (GKD)

•
There is train–inference mismatch causes OOD failures when students 
self-generate.

•
Fix: Student generates; Teacher provides feedback; optimize Reverse 
KL per token.

•
Trains on on-policy samples to align training and deployment behavior.

Source: CMU Lecture slides on Imitation Learning

## --- [Page 118] ---
(C) On-Policy / Generalized KD (GKD)

1. On-policy Data: Sample output sequences from the student

2. Feedback: Run inference using the teacher to get logits on student samples

3. Supervised Training: Minimize mismatch (e.g., KL-divergence) between student and 
teacher token-level logits.

## --- [Page 119] ---
Mode-Covering vs. Mode-Seeking (Why Mix KLs)

•
Forward KL ⇒ mode-covering; Reverse KL ⇒ mode-seeking.

•
Small-capacity students often benefit from some mode-seeking to 
avoid bizarre tokens.

•
Empirically, mixing (e.g., Jeffreys 0.5/0.5) worked best in experiments.

## --- [Page 120] ---
On-Policy Distillation

•
On-Policy GKD outperforms other distillation approaches on T5 models

Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-
Generated Mistakes. ICLR 2024

## --- [Page 121] ---
On-Policy Distillation

•
Qwen3-8B on par with Deepseek-R1.

•
On-policy distillation is 10x more compute efficient than reinforcement 
learning

Agarwal et al. On-Policy Distillation of Language Models: Learning from Self-
Generated Mistakes. ICLR 2024

## --- [Page 122] ---
Symbolic Chain of Thought Distillation

•
small models can also “think” step-by-
step” presented a method that enables 
smaller language models to learn step-
by-step reasoning capabilities

•
A smaller student model is trained on 
CoT samples from a much larger 
teacher model. ( >50B parameters.)

Li et al. Symbolic Chain-of-Thought Distillation: Small Models Can Also 
"Think" Step-by-Step

## --- [Page 123] ---
Symbolic Chain of Thought Distillation

•
Teacher Model: Large language model (e.g., GPT-3 175B)

•
Student Model: Smaller model (e.g., OPT 125M-1.3B)

## --- [Page 124] ---
RL-enhanced distillation, DeepSeek

The teacher model provides not just output probabilities but also rewards 
that help shape the student's behavior

•
Direct RL distillation through DeepSeek-R1-Zero, achieving 71.0% on AIME 2024 
without supervised fine-tuning

•
Hybrid approach with DeepSeek-R1, combining cold-start data with iterative RL 
fine-tuning, reaching 79.8% on AIME 2024

https://labs.adaline.ai/p/llm-distillation-explained

## --- [Page 125] ---
A Scaling Law for Distilled Students

This paper extends that idea to distillation:

•
Student loss 𝐿𝑆is a function of

○
student parameters 𝑁𝑆,

○
distillation tokens 𝐷𝑆,

○
teacher cross-entropy 𝐿𝑇.

•
Teacher size and training tokens matter only through the resulting 
teacher loss 𝐿𝑇.

•
Student loss follows a power law in 𝑁𝑆and 𝐷𝑆 ,and a broken power law 
in 𝐿𝑇

## --- [Page 126] ---
Key Empirical Findings
Teacher Cross-entropy Alone Predicts Student 
Performance

## --- [Page 127] ---
Compute-Optimal Distillation vs. Supervised Training

•
The authors compare four compute scenarios

○
best case (teacher fully amortized),

○
teacher inference only,

○
teacher pretraining only,

○
teacher pretraining plus inference.
•
Given enough compute, supervised training always matches the best possible 
distillation for a fixed student size.
•
Distillation is more compute-efficient than supervised training only when

➢student compute stays below a size-dependent threshold, and

➢a suitable teacher already exists or will be reused for many students.

## --- [Page 128] ---
Why Distillation Scaling Laws Are Important

•
Provides the first large-scale, controlled distillation scaling law for 
language models.

•
Clarifies the capacity gap phenomenon and when stronger teachers 
stop helping students.

•
Gives compute-optimal recipes for

○
choosing teacher strength,

○
allocating tokens between T and S

○
deciding whether to distill or train

## --- [Page 129] ---
Why Distillation Scaling Laws Are Important

•
Practical takeaway:

•
Distillation is best when you

○
have or plan to reuse a strong teacher, and

○
want many capable small models under limited compute.

•
For single models at very large compute, direct supervised training is 
simpler and optimal.

## --- [Page 130] ---
When Distillation Beats Supervised Training

•
Distillation이supervised pretraining보다좋은경우

○
teacher가이미존재할때

○
student에사용할compute나tokens가‘너무많지않을때’

○
student가teacher보다훨씬작아서capacity gap의왼쪽영역에있을때

•
Distillation이손해가되는경우

○
teacher를새로학습해야하고, student는1명뿐일때

○
student compute가충분히클떄

○
teacher가너무강하거나너무약할때

## --- [Page 131] ---
Human-like Communication, Computational Social Science

•
System 2 reasoning distillation for small LLMs

•
Human-in-the-loop, Agents collaborating with human

Perception
Intuition
Reasoning

Automatic Chart

Generation
NAACL 2025

Disambiguating 
Robot Commands

ICRA 2024

NormLens : 
GroundedNorms
EMNLP 2023

Commonsense-
aware Navigation

ICRA 2025

Symbolic CoT

Distillation
ACL 2023

Dialogue CoT

Distillation
EMNLP 2023

## --- [Page 132] ---
[Ongoing] Transfer Knowledge from VLM/VLA to sVLA

•
Vision Language Action model distillation / Lightweighting

•
Evolutionary distillation on simulated environment

Disambiguating 
Robot Commands

ICRA 2024

Commonsense-
aware Navigation

ICRA 2025

Symbolic CoT

Distillation
ACL 2023

Dialogue CoT

Distillation
EMNLP 2023

## --- [Page 133] ---
Distillation for Multimodal Gen AI

VIP : Video Gen

Distillation
ICCV 2025

## --- [Page 134] ---
Conclusion : Distillation, Pros and Cons

Computational Efficiency

•
Up to 90% parameter reduction while preserving core reasoning

•
Large reductions in storage, memory footprint, and deployment-time 
energy usage

Performance Improvements

•
Smoothed KD reduces hallucinations

•
Stronger generalization across 
diverse tasks and domains

## --- [Page 135] ---
Practical Applications

•
Enables real-time inference on mobile and edge devices

•
Reduces infrastructure needs for broad accessibility

•
Cost-effective scaling for production deployment

Limitations & Challenges

•
Hard reasoning tasks still show a performance gap

•
Requires expertise in tuning (temperature, loss balancing)

•
Optimal settings vary significantly by task

Conclusion : Distillation, Pros and Cons

## --- [Page 136] ---
Conclusion : Key Takeaways & What’s Next

•
Distillation is the fastest path to small, fast, affordable LLMs with solid 
quality.

•
Start with synthetic KD; add on-policy GKD if robustness is required.

•
Distillation Scaling Law, Optimal Teacher & Student strategy

•
Inference acceleration (e.g., speculative decoding) and newer KD 
variants.

•
Multimodal Distillation, VLA Distillation are next promising challenge

## --- [Page 137] ---
Reference

•
The Magic of LLM Distillation - Rishabh Agarwal, Google DeepMind

•
Scaling Laws for Autoregressive Generative Modeling 
https://arxiv.org/pdf/2010.14701.pdf

•
Scaling Laws for Neural Language Models https://arxiv.org/pdf/2001.08361.pdf

•
Neural Scaling Laws and GPT-3, 
https://www.youtube.com/watch?v=QMqPAM_knrE

•
Beyond neural scaling laws- data pruning
https://arxiv.org/abs/2206.14486

## --- [Page 138] ---
Thanks for listening!!

youngjaeyu@snu.ac.kr