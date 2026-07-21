# 🧠 Training Large Language Models — The Complete Guide

> **From Pre-Training to RLHF, and why Parameter-Efficient Fine-Tuning (PEFT) changed everything.**
>
> Written for someone who is **brand new** to language models. Every concept builds on the previous one like a story.

---

## 📖 Table of Contents

1. [The Big Picture — What Are We Building?](#1--the-big-picture--what-are-we-building)
2. [Stage 1: Pre-Training — Teaching the Model to Read](#2--stage-1-pre-training--teaching-the-model-to-read)
3. [Stage 2: Supervised Fine-Tuning (SFT) — Teaching It to Talk](#3--stage-2-supervised-fine-tuning-sft--teaching-it-to-talk)
4. [🤔 Wait... SFT Has a Huge Problem](#4---wait-sft-has-a-huge-problem)
5. [The Cost Problem — Why Full Fine-Tuning Is Painful](#5--the-cost-problem--why-full-fine-tuning-is-painful)
6. [PEFT — Parameter-Efficient Fine-Tuning (The Smart Solution)](#6--peft--parameter-efficient-fine-tuning-the-smart-solution)
   - [6.1 LoRA (Low-Rank Adaptation)](#61-lora-low-rank-adaptation)
   - [6.2 QLoRA (Quantized LoRA)](#62-qlora-quantized-lora)
   - [6.3 Adapters](#63-adapters)
   - [6.4 Prefix Tuning](#64-prefix-tuning)
   - [6.5 Prompt Tuning](#65-prompt-tuning)
   - [6.6 PEFT Comparison Table](#66-peft-techniques--comparison-table)
7. [Stage 3: Preference Training — Teaching It What Humans Like](#7--stage-3-preference-training--teaching-it-what-humans-like)
   - [7.1 RLHF (Reinforcement Learning from Human Feedback)](#71-rlhf--reinforcement-learning-from-human-feedback)
   - [7.2 DPO (Direct Preference Optimization)](#72-dpo--direct-preference-optimization)
   - [7.3 RLHF vs DPO Comparison](#73-rlhf-vs-dpo--comparison)
8. [Full Pipeline — Old vs New (Summary)](#8--full-pipeline--old-vs-new-summary)
9. [Tracing One Prompt Through All 3 Stages](#9--tracing-one-prompt-through-all-3-stages)
10. [Glossary](#10--glossary)

---

## 1. 🌍 The Big Picture — What Are We Building?

A **Large Language Model (LLM)** is a computer program that can read and write human language. ChatGPT, Claude, Gemini, LLaMA — these are all LLMs.

But here's the thing: **you don't just "build" an LLM in one step.** Training happens in **stages**, where each stage teaches the model something different.

### The Two Pipelines

```
OLD Pipeline (GPT-2, early GPT-3, BERT):
┌──────────────────┐      ┌──────────────────┐
│  Pre-Training     │ ───▶ │  Fine-Tuning     │ ───▶  Done
│  (learn language) │      │  (SFT)           │
└──────────────────┘      └──────────────────┘

NEW Pipeline (ChatGPT, Claude, Gemini, LLaMA-2-Chat):
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Pre-Training     │ ───▶ │  Fine-Tuning     │ ───▶ │  Preference      │ ───▶  Done
│  (learn language) │      │  (SFT)           │      │  Training        │
└──────────────────┘      └──────────────────┘      │  (RLHF / DPO)   │
                                                     └──────────────────┘
```

Let's walk through each stage. Every section ends with a **natural question** that leads you into the next concept.

---

## 2. 📚 Stage 1: Pre-Training — Teaching the Model to Read

### 🎯 Goal
Teach the model the structure of language, world knowledge, facts, code, and reasoning.

### 📦 Data
Billions of pages of text from the internet — books, Wikipedia, news, code, research papers, forums. Everything.

### ⚙️ Technique: Next-Token Prediction

The model reads text one word at a time and tries to **predict the next word**. It does this **trillions** of times.

```
Input:   "The capital of France is ___"
Target:   "Paris"

Model guesses "Paris"  → ✅ Correct! Reinforce these weights.
Model guesses "London" → ❌ Wrong! Adjust weights to do better next time.

Input:   "import numpy as ___"
Target:   "np"

Input:   "Water boils at 100 degrees ___"
Target:   "Celsius"
```

After each wrong guess, the model adjusts its internal numbers (called **weights** or **parameters**) slightly. After trillions of guesses, it becomes incredibly good at language.

### 📊 Scale of Pre-Training

| What                | How Much                              |
|---------------------|---------------------------------------|
| Training data       | 1 – 15+ **trillion** tokens           |
| Model size          | Millions to **hundreds of billions** of parameters |
| Hardware            | Thousands of GPUs, running for **weeks to months** |
| Cost                | **$1 million to $100+ million** per run |

### 🧪 What Does a Pre-Trained Model Look Like?

It knows language, facts, and reasoning — but it behaves like a **text-completion engine**, not an assistant:

```
You type:    "What is photosynthesis?"

Model output: "What is photosynthesis? This is a common question 
              in biology. Chapter 5 of Campbell Biology covers the 
              light-dependent reactions in detail. The process was 
              first studied by Jan Ingenhousz in 1779..."

              ❌ It continues the TEXT, like writing a Wikipedia article.
              ❌ It does NOT answer YOUR question.
```

> ### 🤔 Question: "This model knows everything but can't hold a conversation. How do we fix that?"
>
> **Answer: We teach it to follow instructions. That's Stage 2 — Supervised Fine-Tuning (SFT).**

---

## 3. 🎓 Stage 2: Supervised Fine-Tuning (SFT) — Teaching It to Talk

### 🎯 Goal
Transform the text-completion engine into a helpful **assistant** that follows instructions.

### 📦 Data
Thousands of **(prompt, ideal response)** pairs, written by humans.

### ⚙️ Technique: Supervised Learning on Demonstrations

Human annotators write thousands of example conversations. The model is trained to mimic these ideal responses.

### 📝 Example SFT Training Data

| User Prompt | Ideal Response (Human-Written) |
|---|---|
| What is the capital of Japan? | The capital of Japan is Tokyo. |
| Explain gravity to a 5-year-old. | You know how when you throw a ball up, it always comes back down? That's gravity! The Earth is really big, so it pulls everything toward it. |
| Write a Python function to add two numbers. | `def add(a, b): return a + b` |
| Summarize the French Revolution in 2 sentences. | The French Revolution (1789–1799) was a period of radical political upheaval in France. It ended the monarchy and spread democratic ideals across Europe. |
| Translate "good morning" to Spanish. | "Buenos días" |

### ✅ What SFT Teaches the Model

- **Answer questions directly** (don't just continue text)
- **Follow instructions** (format, length, tone)
- **Be helpful and polite**
- **Produce structured responses** (lists, code, summaries)

### 🧪 After SFT — The Model Can Chat!

```
You type:    "What is photosynthesis?"

Model output: "Photosynthesis is the process by which plants convert 
              sunlight, water, and carbon dioxide into glucose and 
              oxygen. It takes place in the chloroplasts of plant cells."

              ✅ It answers your question!
              ✅ It's concise and helpful!
```

> ### 🤔 Question: "Great, it can talk now. But is every answer equally good? Can it sometimes give bad answers?"
>
> **Answer: Yes! That's the big limitation of SFT alone...**

---

## 4. 🤔 Wait... SFT Has a Huge Problem

SFT teaches the model **WHAT** to do (follow instructions), but not **WHICH** answer is best when there are multiple reasonable options.

### Example: The Same Prompt, Three Different Answers

```
User: "Is coffee good for your health?"

Answer A: "Yes."
          ❌ Too brief. Not helpful.

Answer B: "Coffee has both benefits and risks. Moderate consumption 
           (2-3 cups/day) is linked to lower risk of type 2 diabetes 
           and Parkinson's. However, excessive intake can cause 
           anxiety and sleep issues. It depends on the individual."
          ✅ Balanced, nuanced, helpful.

Answer C: "NEVER drink coffee! It destroys your body!!!"
          ❌ Alarmist. Misleading. Wrong.
```

**The SFT model can produce ANY of these.** There is **no signal** telling it that Answer B is the best one.

> ### 🤔 Question: "OK, so we need a way to teach the model which answers humans prefer. But first... SFT required updating ALL the model's parameters. For a 175-billion parameter model, isn't that insanely expensive?"
>
> **Answer: Yes. It is. And that leads us to a critical problem...**

---

## 5. 💰 The Cost Problem — Why Full Fine-Tuning Is Painful

Let's think about this with real numbers.

### The Math of Full Fine-Tuning

When we do SFT (or any fine-tuning), we update the model's **parameters** (weights). Here's what that looks like for real models:

| Model | Parameters | Memory to Store Model | Memory for Training (with gradients + optimizer) | GPU Cost |
|---|---|---|---|---|
| GPT-2 | 1.5 billion | ~3 GB | ~18 GB | 1 GPU, hours |
| LLaMA-7B | 7 billion | ~14 GB | ~84 GB | 2-4 GPUs, days |
| LLaMA-70B | 70 billion | ~140 GB | ~840 GB | 16-32 GPUs, days |
| GPT-3 | 175 billion | ~350 GB | ~2,100 GB | 64+ GPUs, weeks |

### Why Does Training Use 6x the Model Size in Memory?

```
For each parameter, you need to store:
   1. The parameter itself            → 2 bytes  (float16)
   2. Its gradient                    → 2 bytes  (float16)
   3. Optimizer state (Adam has 2)    → 8 bytes  (float32)
                                        ─────────
                              Total:    12 bytes per parameter

GPT-3: 175 billion × 12 bytes = ~2,100 GB = ~2.1 TB of GPU memory!

A single A100 GPU has 80 GB of memory.
You need ~26 A100 GPUs JUST to fit the training state.
At ~$2/hour per GPU, that's $52/hour JUST for memory.
A typical SFT run takes days → $5,000 – $50,000+
```

### The Questions That Changed Everything

> **🤔 "Do we really need to update ALL 175 billion parameters just to teach the model to be a chatbot?"**
>
> **🤔 "The model already knows language, facts, and reasoning from pre-training. SFT just teaches it a new *style*. Can we do that with fewer changes?"**
>
> **🤔 "What if we could freeze most of the model and only train a tiny fraction of the parameters?"**
>
> **Answer: YES. This is exactly what PEFT does.**

---

## 6. 🔧 PEFT — Parameter-Efficient Fine-Tuning (The Smart Solution)

### The Core Idea

Instead of updating **all** parameters during fine-tuning, **freeze most of the model** and only train a **small number of new or modified parameters.**

```
Full Fine-Tuning:
┌─────────────────────────────────────────────┐
│  ALL 175 billion parameters get updated     │  → Expensive!
│  ██████████████████████████████████████████  │  → $50,000+
└─────────────────────────────────────────────┘

PEFT (e.g., LoRA):
┌─────────────────────────────────────────────┐
│  175 billion parameters FROZEN              │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  Only ~10 million NEW parameters trained    │  → Cheap!
│  ████                                       │  → $100–$500
└─────────────────────────────────────────────┘
       ▲
       Only 0.01% of the model is trained!
```

### Why Does PEFT Work?

Research showed a surprising finding: when you fine-tune a large model, the weight changes live in a **low-dimensional subspace**. In simpler terms, the updates have a lot of **redundancy**. You don't need to change all parameters — a small, targeted set of changes is enough to shift the model's behavior.

Now let's look at the specific PEFT techniques:

---

### 6.1 LoRA (Low-Rank Adaptation)

> **The most popular PEFT technique. Used everywhere today.**

**Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

#### The Idea

Instead of modifying a huge weight matrix directly, LoRA adds a **small detour** (two tiny matrices) beside it.

```
Original (Full Fine-Tuning):
                ┌──────────────┐
  Input ───────▶│  W (huge)    │───────▶ Output
                │  d × d       │
                │  (updated)   │
                └──────────────┘
                
                W is a matrix of, say, 4096 × 4096 = 16.7 million numbers
                During fine-tuning, ALL 16.7M numbers get updated.


LoRA (Efficient Fine-Tuning):
                ┌──────────────┐
  Input ───────▶│  W (frozen)  │──┐
                │  d × d       │  │
                │  (NO update) │  ├──── Add ──▶ Output
                └──────────────┘  │
                                  │
                ┌─────┐  ┌─────┐  │
  Input ───────▶│A    │─▶│B    │──┘
                │d × r│  │r × d│
                └─────┘  └─────┘
                
                r = "rank" = typically 8, 16, or 64
                
                If d = 4096 and r = 8:
                  Original W:  4096 × 4096 = 16,777,216 parameters
                  LoRA A + B:  (4096 × 8) + (8 × 4096) = 65,536 parameters
                  
                  That's only 0.39% of the original!
```

#### LoRA Example with Real Numbers

```
Model: LLaMA-7B (7 billion parameters)

Full Fine-Tuning:
  - Parameters trained:  7,000,000,000  (7 billion)
  - GPU memory needed:   ~84 GB
  - Hardware:            4× A100 80GB GPUs
  - Cost:               ~$2,000 – $10,000
  - Training time:       2-5 days

LoRA Fine-Tuning (rank=8):
  - Parameters trained:  ~10,000,000  (10 million) = 0.14% of model
  - GPU memory needed:   ~16 GB
  - Hardware:            1× A100 or even 1× RTX 4090
  - Cost:               ~$50 – $200
  - Training time:       A few hours

Same quality. 50× cheaper. 🤯
```

---

### 6.2 QLoRA (Quantized LoRA)

> **🤔 "LoRA is great, but I still need to LOAD the entire frozen model into GPU memory. For a 70B model, that's 140 GB. Can we shrink that too?"**
>
> **Answer: Yes! Compress (quantize) the frozen model to use less memory.**

**Paper:** "QLoRA: Efficient Finetuning of Quantized Language Models" (Dettmers et al., 2023)

#### The Idea

1. **Quantize** the frozen base model from 16-bit to **4-bit** (shrinks memory by 4×)
2. Apply **LoRA** adapters on top (trained in 16-bit for accuracy)

```
Normal LoRA:
  Base model in 16-bit:  LLaMA-65B → ~130 GB  (need multiple GPUs)
  + LoRA adapters:       ~40 MB

QLoRA:
  Base model in 4-bit:   LLaMA-65B → ~33 GB   (fits on 1 GPU! 🎉)
  + LoRA adapters:       ~40 MB (still 16-bit for quality)
```

#### QLoRA Real-World Impact

```
Fine-tuning LLaMA-65B:

Full Fine-Tuning:        ~780 GB GPU memory   → 16+ A100 GPUs  → $20,000+
LoRA:                    ~130 GB GPU memory   → 2× A100 GPUs   → $1,000
QLoRA:                   ~33 GB GPU memory    → 1× A100 GPU    → $200

QLoRA made it possible to fine-tune a 65-BILLION parameter model
on a SINGLE consumer GPU. This democratized LLM fine-tuning.
```

---

### 6.3 Adapters

> **An earlier PEFT approach (2019), before LoRA existed.**

**Paper:** "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al., 2019)

#### The Idea

Insert small **adapter modules** (tiny neural networks) between the existing layers of the model. Freeze the original model; only train the adapters.

```
Original Transformer Layer:
  Input → [Attention] → [Feed-Forward] → Output

With Adapters:
  Input → [Attention] → [🔧 Adapter] → [Feed-Forward] → [🔧 Adapter] → Output
                          (trainable)                      (trainable)

Each adapter is tiny:
  Down-project:  d → bottleneck  (e.g., 4096 → 64)
  Non-linearity: ReLU
  Up-project:    bottleneck → d  (e.g., 64 → 4096)
  Residual connection (add input back)
  
  Parameters per adapter: 2 × d × bottleneck = 2 × 4096 × 64 = 524,288
  For 32 layers × 2 adapters each: ~33 million parameters (vs 7 billion)
```

#### Adapters vs LoRA

| Aspect | Adapters | LoRA |
|---|---|---|
| Where added | Between layers (sequential) | Beside weight matrices (parallel) |
| Inference speed | Slightly slower (extra layers to compute) | **No slowdown** (merge weights after training) |
| Popularity today | Less common | **Dominant** |

> LoRA "won" because you can **merge** the LoRA weights back into the original model after training, so there's **zero extra cost at inference time.** Adapters always add a small overhead.

---

### 6.4 Prefix Tuning

> **🤔 "What if instead of modifying the model at all, we just add some special learnable tokens to the input?"**

**Paper:** "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (Li & Liang, 2021)

#### The Idea

Prepend a sequence of **learnable virtual tokens** (called a "prefix") to the input at every layer of the transformer. These tokens are not real words — they're continuous vectors that the model learns during training.

```
Normal input to each layer:
  [user's actual tokens...]

With Prefix Tuning:
  [PREFIX_1, PREFIX_2, ..., PREFIX_k, user's actual tokens...]
   ▲ these are trainable           ▲ these are processed normally
   ▲ continuous vectors, not real words

Typical prefix length: 10-100 virtual tokens
Parameters trained: prefix_length × num_layers × hidden_dim × 2
                    = 20 × 32 × 4096 × 2 ≈ 5.2 million (for a 7B model)
```

#### When to Use Prefix Tuning

- When you want to keep the model **completely untouched**
- When you need **multiple task-specific prefixes** (just swap the prefix, same model)
- Works better for **generation tasks** (summarization, translation)

---

### 6.5 Prompt Tuning

> **An even simpler version of Prefix Tuning.**

**Paper:** "The Power of Scale for Parameter-Efficient Prompt Tuning" (Lester et al., 2021)

#### The Idea

Same concept as prefix tuning, but the learnable tokens are only added to the **input layer** (not every layer).

```
Prefix Tuning:    adds virtual tokens at EVERY layer   → more parameters
Prompt Tuning:    adds virtual tokens at INPUT only     → fewer parameters

Prompt Tuning parameters: prefix_length × hidden_dim
                          = 20 × 4096 = 81,920 parameters

That's 0.001% of a 7B model! Incredibly tiny.
```

#### Trade-off

- **Very few parameters** → fastest to train, smallest storage
- **Slightly lower quality** than LoRA on smaller models
- **Competitive with full fine-tuning** on very large models (10B+)

---

### 6.6 PEFT Techniques — Comparison Table

| Technique | Year | Parameters Trained | Memory Saving | Inference Overhead | Quality vs Full FT | Best For |
|---|---|---|---|---|---|---|
| **Full Fine-Tuning** | — | 100% | None | None | Baseline | Unlimited budget |
| **Adapters** | 2019 | ~0.5–4% | Moderate | Small overhead | ~95-98% | Early PEFT research |
| **Prefix Tuning** | 2021 | ~0.1% | High | Slightly longer context | ~90-95% | Generation tasks, multi-task |
| **Prompt Tuning** | 2021 | ~0.001% | Very high | Minimal | ~85-95% (scales with model size) | Very large models, many tasks |
| **LoRA** | 2021 | ~0.1–1% | High | **None** (merge weights) | ~97-100% | **Most popular. General purpose.** |
| **QLoRA** | 2023 | ~0.1–1% | **Very high** (4-bit base) | None (merge weights) | ~95-99% | **Low-resource. Consumer GPUs.** |

### 💡 Key Insight: How to Choose

```
Decision Flow:

Do you have unlimited GPUs and budget?
  └─ YES → Full Fine-Tuning (best quality, simplest)
  └─ NO  ↓

Do you have at least 1–2 A100 GPUs (80GB)?
  └─ YES → LoRA (best quality-to-cost ratio)
  └─ NO  ↓

Do you only have a consumer GPU (RTX 3090/4090, 24GB)?
  └─ YES → QLoRA (makes large models accessible)
  └─ NO  ↓

Do you only have a very small GPU or want extreme efficiency?
  └─ YES → Prompt Tuning (tiniest footprint)
```

> ### 🤔 Question: "OK, now I know how to efficiently fine-tune a model (SFT with PEFT). But we still have that problem from Section 4 — the model doesn't know which answers humans *prefer*. How do we fix that?"
>
> **Answer: That's Stage 3 — Preference Training. Let's go.**

---

## 7. 🏆 Stage 3: Preference Training — Teaching It What Humans Like

This is the **critical third stage** that separates modern models (ChatGPT, Claude) from older ones (GPT-2). The model has learned language (Stage 1) and how to follow instructions (Stage 2). Now we teach it **which answers are better**.

### The General Process

```
Step 1:  Model generates TWO answers to the same prompt
Step 2:  A human reviewer picks the BETTER answer
Step 3:  Model learns: "produce more like the preferred one, 
         less like the rejected one"
Step 4:  Repeat thousands of times
```

### Example Preference Data

```
Prompt: "How do I deal with a difficult coworker?"

Preferred ✅: "Here are some strategies that might help:
  First, try to understand their perspective — they might be 
  dealing with stress you're not aware of. Direct, private 
  conversation often works best. Use 'I' statements like 
  'I feel frustrated when...' If the situation doesn't improve, 
  consider involving HR or a manager for mediation."

Rejected ❌: "Just ignore them. Or better yet, be difficult 
  back. Give them a taste of their own medicine. If that 
  doesn't work, go to your boss and get them fired."

Human says: First one is better → The model learns from this.
```

There are two main techniques for this stage:

---

### 7.1 RLHF — Reinforcement Learning from Human Feedback

> **The original technique. Used to train ChatGPT and early Claude.**

**Paper:** "Training language models to follow instructions with human feedback" (Ouyang et al., 2022)

#### RLHF Has Two Sub-Steps

**Sub-Step A: Train a Reward Model**

A separate, smaller neural network that learns to predict human preferences.

```
Training the Reward Model:

Input:  (prompt, response) pair
Output: a SCORE (higher = humans would prefer this)

Example:
  Prompt: "Explain quantum computing"
  
  Response A: "Quantum computing uses qubits that can      → Score: 8.5
    exist in multiple states simultaneously, unlike 
    classical bits. This allows quantum computers to 
    solve certain problems exponentially faster..."
  
  Response B: "Quantum computing is just hype. Regular     → Score: 2.1
    computers are fine for everything."
  
The Reward Model is trained on thousands of these pairs until
it can reliably predict which response a human would prefer.
```

**Sub-Step B: Optimize the LLM using PPO**

PPO (Proximal Policy Optimization) is a reinforcement learning algorithm.

```
The RLHF Training Loop:

┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. Take a prompt from the dataset                  │
│  2. LLM generates a response                        │
│  3. Reward Model scores the response (e.g., 7.4)    │
│  4. PPO updates LLM to get HIGHER scores            │
│  5. KL penalty keeps LLM close to SFT version       │
│     (prevents "reward hacking")                     │
│  6. Repeat millions of times                        │
│                                                     │
└─────────────────────────────────────────────────────┘

What is "reward hacking"?
  Without the KL penalty, the model might learn tricks like:
  - Being excessively verbose (longer = higher score)
  - Excessive flattery ("What a GREAT question!")
  - Repeating the same safe phrases over and over
  
  The KL penalty says: "improve, but don't deviate too far 
  from the original SFT model."
```

#### RLHF Summary

| Component | What It Is |
|---|---|
| Reward Model | Separate NN that scores responses |
| PPO | RL algorithm that updates the LLM |
| KL Penalty | Prevents reward hacking |
| Models in memory | **3** (LLM + Reward Model + Reference LLM) |
| Complexity | High — many moving parts |

---

### 7.2 DPO — Direct Preference Optimization

> **🤔 "RLHF works, but it needs a separate Reward Model, a complex PPO loop, and 3 models in memory. Can we do this more simply?"**
>
> **Answer: Yes. DPO skips the Reward Model entirely.**

**Paper:** "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model" (Rafailov et al., 2023)

#### The Key Insight

The DPO authors proved mathematically that **you don't need a separate Reward Model**. The LLM itself can be treated as an implicit reward model. Instead of the complex RL loop, you can directly train the LLM on preference pairs using a simple supervised loss function.

```
RLHF (complex):
  Preference Data → Train Reward Model → RL Loop (PPO) → Updated LLM
  (3 models in memory, unstable, expensive)

DPO (simple):
  Preference Data → Direct Loss Function → Updated LLM
  (2 models in memory, stable, cheaper)
```

#### How DPO Works

```
For each training example:

  Prompt:     "How do I lose weight?"
  Preferred:  "A sustainable approach combines balanced eating with 
               regular exercise. Aim for a moderate deficit of 
               300-500 cal/day..."
  Rejected:   "Just stop eating. A 7-day water fast works fastest."

DPO Loss Function does two things simultaneously:
  1. INCREASE the probability of generating the preferred response
  2. DECREASE the probability of generating the rejected response

That's it. No reward model. No PPO. No RL loop.
Just a straightforward optimization on pairs of examples.
```

#### DPO Advantages

```
✅ No separate Reward Model needed (saves memory + compute)
✅ No unstable RL training loop
✅ Simpler to implement (~20 lines of PyTorch vs hundreds for RLHF)
✅ More stable training
✅ Comparable or better results on many benchmarks
```

---

### 7.3 RLHF vs DPO — Comparison

| Aspect | RLHF | DPO |
|---|---|---|
| **Year** | 2017 (popularized 2022) | 2023 |
| **Reward Model** | Yes — separate model | No — LLM is its own reward model |
| **RL Algorithm** | PPO (complex) | None (supervised loss) |
| **Models in Memory** | 3 (LLM + Reward Model + Reference) | 2 (LLM + Reference) |
| **Compute Cost** | High | Lower |
| **Stability** | Can be unstable | More stable |
| **Implementation** | Hundreds of lines, many hyperparameters | ~20 lines, fewer hyperparameters |
| **Used By** | ChatGPT (early), InstructGPT | LLaMA-2, Zephyr, many open-source |
| **Quality** | Excellent at scale | Comparable, sometimes better |

> ### 💡 Note: PEFT Applies Here Too!
> 
> Just like SFT, preference training (RLHF/DPO) also updates model weights — so **LoRA and QLoRA can be used in Stage 3 as well**, making preference training much cheaper. This combination is sometimes called **"LoRA + DPO"** and is the most popular approach in open-source LLM alignment.

---

## 8. 🗺️ Full Pipeline — Old vs New (Summary)

```
════════════════════════════════════════════════════════════════════
  OLD PIPELINE (GPT-2, early GPT-3 era)
════════════════════════════════════════════════════════════════════

  ┌────────────────┐         ┌──────────────────┐
  │  PRE-TRAINING  │ ──────▶ │  SFT             │ ──────▶ Model Ready
  │                │         │  (Full FT or      │
  │  Next-token    │         │   PEFT like LoRA) │
  │  prediction    │         │                   │
  └────────────────┘         └──────────────────┘
     Trillions of              Thousands of
     tokens                    (prompt, response) pairs


════════════════════════════════════════════════════════════════════
  NEW PIPELINE (ChatGPT, Claude, Gemini, LLaMA-2-Chat)
════════════════════════════════════════════════════════════════════

  ┌────────────────┐    ┌──────────────┐    ┌───────────────────┐
  │  PRE-TRAINING  │──▶ │  SFT         │──▶ │  PREFERENCE       │──▶ Model
  │                │    │              │    │  TRAINING          │    Ready
  │  Next-token    │    │  Full FT     │    │                   │
  │  prediction    │    │  or LoRA     │    │  RLHF or DPO      │
  │                │    │  or QLoRA    │    │  (also with LoRA!) │
  └────────────────┘    └──────────────┘    └───────────────────┘
     Trillions of         Thousands of        Thousands of
     tokens               demonstrations      preference pairs
```

### The Progression of What Each Stage Teaches

| Stage | What It Teaches | Analogy |
|---|---|---|
| **Pre-Training** | Language, knowledge, reasoning | A student reads every book in the library |
| **SFT** | How to follow instructions | A tutor shows the student how to answer questions properly |
| **Preference Training** | Which answers humans prefer | A mentor says "This answer is better than that one — here's why" |

---

## 9. 🔬 Tracing One Prompt Through All 3 Stages

**Prompt:** `"Should I invest my savings in cryptocurrency?"`

### After Stage 1 — Pre-Training Only

```
"Should I invest my savings in cryptocurrency? This question 
has been debated extensively since Bitcoin's creation in 2009 
by the pseudonymous Satoshi Nakamoto. The cryptocurrency 
market has experienced significant volatility, with Bitcoin 
reaching an all-time high of $69,000 in November 2021 before 
declining to..."

❌ Writes an article. Does NOT answer you.
```

### After Stage 2 — Pre-Training + SFT

```
"Cryptocurrency can be a good investment! Bitcoin and Ethereum 
have shown strong returns over the past decade. I'd recommend 
putting 10-20% of your portfolio into crypto. Start with 
Bitcoin as a core holding."

✅ Answers the question!
❌ Overconfident. No disclaimer.
❌ Gives risky financial advice.
❌ Doesn't mention volatility or risks.
```

### After Stage 3 — Pre-Training + SFT + Preference Training

```
"That's an important financial decision that depends on your 
personal situation. Here are some things to consider:

  • Crypto is highly volatile — prices can swing 20-50% in days
  • Only invest what you can afford to lose entirely
  • Diversification across asset classes helps manage risk
  • Consider your risk tolerance, time horizon, and emergency fund

I'd recommend speaking with a qualified financial advisor who 
can assess your specific circumstances before making investment 
decisions. I can help you think through the factors, but I'm 
not able to give personalized financial advice."

✅ Balanced and honest
✅ Acknowledges uncertainty
✅ Lists real risks
✅ Suggests professional help
✅ Discloses its limitations
```

---

## 10. 📚 Glossary

| Term | Definition |
|---|---|
| **Token** | The smallest unit a model reads. Roughly 1 token ≈ ¾ of a word. "Photosynthesis" = 4 tokens. |
| **Parameters / Weights** | The internal numbers the model adjusts during training. GPT-3 has 175B; GPT-4 is estimated at over 1 trillion. |
| **Pre-Training** | Stage 1 — the model learns language by predicting the next token across trillions of words. |
| **SFT** | Supervised Fine-Tuning. Stage 2 — the model learns to follow instructions from human-written examples. |
| **Full Fine-Tuning** | Updating ALL parameters of the model. Expensive but thorough. |
| **PEFT** | Parameter-Efficient Fine-Tuning. A family of techniques that freeze most parameters and only train a small fraction. |
| **LoRA** | Low-Rank Adaptation. Adds small trainable matrices beside frozen weight matrices. Most popular PEFT method. |
| **QLoRA** | Quantized LoRA. Compresses the frozen model to 4-bit, then applies LoRA. Runs on consumer GPUs. |
| **Adapters** | Small neural networks inserted between existing layers. An early PEFT technique. |
| **Prefix Tuning** | Learnable virtual tokens prepended at every layer. |
| **Prompt Tuning** | Learnable virtual tokens prepended at the input layer only. |
| **RLHF** | Reinforcement Learning from Human Feedback. Stage 3 option using a Reward Model + PPO. |
| **DPO** | Direct Preference Optimization. Stage 3 option that trains directly on preference pairs. Simpler than RLHF. |
| **Reward Model** | A separate neural network that scores responses (used in RLHF). |
| **PPO** | Proximal Policy Optimization. The RL algorithm used in RLHF. |
| **KL Divergence** | Measures how different the updated model is from the original. Used as a penalty in RLHF. |
| **Quantization** | Compressing model weights to lower precision (e.g., 16-bit → 4-bit) to reduce memory usage. |
| **Hallucination** | When a model confidently generates false information. Preference training helps reduce this. |
| **Alignment** | Making AI systems behave in ways that are helpful, honest, and harmless. |
| **Reward Hacking** | When the model finds tricks to get high reward scores without being genuinely helpful. |

---

## 🔗 Key References

| Paper | Year | Contribution |
|---|---|---|
| Attention Is All You Need (Vaswani et al.) | 2017 | Invented the Transformer architecture |
| BERT (Devlin et al.) | 2018 | Demonstrated pre-training + fine-tuning paradigm |
| GPT-2 (Radford et al.) | 2019 | Showed large-scale autoregressive pre-training works |
| Adapters (Houlsby et al.) | 2019 | First popular PEFT technique |
| GPT-3 (Brown et al.) | 2020 | 175B parameters, showed few-shot learning |
| Prefix Tuning (Li & Liang) | 2021 | PEFT via learnable prefix tokens |
| Prompt Tuning (Lester et al.) | 2021 | Simplified prefix tuning (input layer only) |
| LoRA (Hu et al.) | 2021 | Low-rank adaptation — dominant PEFT method |
| InstructGPT / RLHF (Ouyang et al.) | 2022 | RLHF for instruction following |
| QLoRA (Dettmers et al.) | 2023 | 4-bit quantization + LoRA for consumer GPUs |
| DPO (Rafailov et al.) | 2023 | Simpler alternative to RLHF |

---

> **💡 TL;DR:** Old models were trained in 2 steps (pre-train → SFT). New models add a 3rd step (preference training) that makes them safer and more aligned with what humans want. PEFT techniques like LoRA and QLoRA made fine-tuning affordable for everyone, not just big tech companies. The field is moving fast — but these fundamentals are the foundation everything else builds on.

---

*Last updated: May 2026*
