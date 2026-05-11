<h1 align="center">
  🧪 RAGAS TestsetGenerator
  <br/>
  <sub>How Synthetic Test Data Generation Works Under the Hood</sub>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/ragas-0.4.x-blue?style=for-the-badge" alt="ragas version"/>
  <img src="https://img.shields.io/badge/langchain-0.3.x-green?style=for-the-badge" alt="langchain version"/>
  <img src="https://img.shields.io/badge/python-3.11-yellow?style=for-the-badge" alt="python version"/>
</p>

---

## 🌐 Overview

RAGAS **TestsetGenerator** automatically creates question-answer pairs from your documents to evaluate RAG (Retrieval Augmented Generation) pipelines. Instead of manually writing test questions, it uses LLMs to generate diverse, high-quality questions at different difficulty levels.

```
 Your Documents ──▶ TestsetGenerator ──▶ Questions + Ground Truth Answers
                                              │
                                              ▼
                                     Evaluate Your RAG Pipeline
```

**Why it matters:** You can't improve what you can't measure. The generated testset gives you a benchmark to score your RAG pipeline on metrics like faithfulness, answer relevancy, and context precision.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAGAS TestsetGenerator                          │
│                                                                     │
│  ┌───────────┐    ┌──────────────┐    ┌───────────┐    ┌────────┐  │
│  │           │    │              │    │           │    │        │  │
│  │  STAGE 1  │───▶│   STAGE 2    │───▶│  STAGE 3  │───▶│STAGE 4 │  │
│  │           │    │              │    │           │    │        │  │
│  │ Document  │    │  Knowledge   │    │  Query    │    │ Critic │  │
│  │ Chunking  │    │  Graph       │    │ Synthesis │    │ Review │  │
│  │           │    │              │    │           │    │        │  │
│  └───────────┘    └──────────────┘    └───────────┘    └────────┘  │
│       │                 │                   │               │      │
│       │                 │                   │               │      │
│  Text Splitter    Embedding Model     Generator LLM    Critic LLM  │
│                                                                     │
│                              ┌──────────┐                           │
│                              │ STAGE 5  │                           │
│                              │ TestSet  │──▶ DataFrame              │
│                              │ Output   │                           │
│                              └──────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📄 Stage 1 — Document Chunking

> **Goal:** Break large documents into smaller, overlapping pieces for processing.

```
┌──────────────────────────────────────────────────────────────┐
│                    📄 Original Document                       │
│                                                              │
│  Paul Graham's essay — 10,000 words                          │
│  "I wrote my first program on an IBM 1401, in the basement   │
│   of our junior high school..."                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              │
                     RecursiveCharacterTextSplitter
                      chunk_size=1024, overlap=20
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Chunk 1  │   │ Chunk 2  │   │ Chunk 3  │   ...N chunks
        │          │   │          │   │          │
        │ 1024     │   │ 1024     │   │ 1024     │
        │ chars    │   │ chars    │   │ chars    │
        └──────────┘   └──────────┘   └──────────┘
              │    ╲         ╱    │
              │     20 char      │
              │     overlap      │
              │                  │
              ▼                  ▼
       No information     Continuity
       lost at borders    preserved
```

**Key Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `chunk_size` | 1024 | Max characters per chunk |
| `chunk_overlap` | 20 | Shared characters between adjacent chunks |

> 💡 **Why overlap?** A sentence at the boundary of two chunks might get cut in half. Overlap ensures both chunks have the complete sentence.

---

## 🧠 Stage 2 — Knowledge Graph Construction

> **Goal:** Understand relationships between chunks using vector embeddings.

```
                        ┌──────────────────────┐
                        │   Embedding Model     │
                        │   (text-embedding-    │
                        │    3-small)           │
                        └──────────┬───────────┘
                                   │
                     Each chunk ──▶ vector
                                   │
                                   ▼

   Chunk 1  ──▶  [0.12, 0.45, 0.78, 0.33, ...]
   Chunk 2  ──▶  [0.15, 0.42, 0.80, 0.31, ...]   ← Similar to Chunk 1
   Chunk 3  ──▶  [0.67, 0.11, 0.23, 0.89, ...]
   Chunk 4  ──▶  [0.65, 0.13, 0.25, 0.87, ...]   ← Similar to Chunk 3

                                   │
                     Cosine similarity scoring
                                   │
                                   ▼

                  ┌─── Knowledge Graph ───┐
                  │                       │
                  │  (C1)───0.95───(C2)   │   High similarity
                  │   │                   │   = same topic
                  │  0.30                 │
                  │   │                   │   Low similarity
                  │  (C3)───0.92───(C4)   │   = different topic
                  │                       │
                  └───────────────────────┘
```

**What the Knowledge Graph enables:**

| Capability | Description |
|-----------|-------------|
| **Topic Clustering** | Groups chunks discussing the same subject |
| **Multi-Context Linking** | Identifies chunks that can be combined for complex questions |
| **Relevance Mapping** | Maps which chunks are most informative |

---

## ❓ Stage 3 — Query Synthesis & Evolution

> **Goal:** Generate diverse questions at varying difficulty levels.

The generator LLM creates three types of questions:

### 🟢 Type 1: Simple Questions

Direct, factual questions from a **single chunk**.

```
┌────────────────────────────────┐
│         Single Chunk           │
│                                │
│  "Paul Graham studied          │
│   philosophy at Cornell        │
│   before switching to AI       │
│   at Harvard."                 │
└───────────────┬────────────────┘
                │
                ▼
       ┌─────────────────┐
       │  Generator LLM  │
       │                  │
       │  Prompt:         │
       │  "Create a       │
       │   straightforward│
       │   factual        │
       │   question"      │
       └────────┬────────┘
                │
                ▼
  ╔══════════════════════════════╗
  ║  "What did Paul Graham      ║
  ║   study at Cornell?"        ║
  ║                             ║
  ║  Difficulty: ⭐              ║
  ║  Type: SIMPLE               ║
  ╚══════════════════════════════╝
```

---

### 🟡 Type 2: Reasoning Questions

Questions requiring **inference and multi-step thinking** from a single chunk.

```
┌────────────────────────────────┐
│         Single Chunk           │
│                                │
│  "Paul Graham's early          │
│   experience with Lisp shaped  │
│   his belief that programming  │
│   languages influence how      │
│   people think about           │
│   problems..."                 │
└───────────────┬────────────────┘
                │
                ▼
       ┌─────────────────┐
       │  Generator LLM  │
       │                  │
       │  Prompt:         │
       │  "Create a       │
       │   question that  │
       │   requires       │
       │   reasoning and  │
       │   inference"     │
       └────────┬────────┘
                │
                ▼
  ╔══════════════════════════════════╗
  ║  "How did Paul Graham's Lisp    ║
  ║   experience influence his      ║
  ║   philosophy on startups?"      ║
  ║                                 ║
  ║  Difficulty: ⭐⭐⭐               ║
  ║  Type: REASONING                ║
  ╚══════════════════════════════════╝
```

---

### 🔴 Type 3: Multi-Context Questions

Questions requiring **information from multiple chunks** to answer.

```
┌──────────────────┐              ┌──────────────────┐
│    Chunk 1       │              │    Chunk 7       │
│                  │              │                  │
│ "Graham saw      │              │ "At YC, Graham   │
│  painting as a   │              │  noticed the     │
│  way to explore  │              │  best founders   │
│  visual ideas"   │              │  were like       │
│                  │              │  artists..."     │
└────────┬─────────┘              └────────┬─────────┘
         │                                 │
         │    Knowledge Graph said:        │
         │    "These chunks are related"   │
         │                                 │
         └──────────┬──────────────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  Generator LLM  │
           │                  │
           │  Prompt:         │
           │  "Create a       │
           │   question       │
           │   requiring BOTH │
           │   chunks"        │
           └────────┬────────┘
                    │
                    ▼
  ╔═══════════════════════════════════════╗
  ║  "Compare Paul Graham's views on     ║
  ║   painting with his observations     ║
  ║   about startup founders at YC."     ║
  ║                                      ║
  ║  Difficulty: ⭐⭐⭐⭐                   ║
  ║  Type: MULTI_CONTEXT                 ║
  ╚═══════════════════════════════════════╝
```

### Evolution Summary

| Type | Chunks Used | Difficulty | Tests |
|------|-------------|-----------|-------|
| 🟢 Simple | 1 | ⭐ | Basic retrieval |
| 🟡 Reasoning | 1 | ⭐⭐⭐ | Inference ability |
| 🔴 Multi-Context | 2+ | ⭐⭐⭐⭐ | Cross-chunk synthesis |

---

## 🔍 Stage 4 — Critic Review & Answer Generation

> **Goal:** Generate ground truth answers and filter out low-quality questions.

```
         ┌──────────────────┐     ┌──────────────────┐
         │    Generated     │     │    Relevant       │
         │    Question      │     │    Context Chunks │
         └────────┬─────────┘     └────────┬──────────┘
                  │                         │
                  └────────┬────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │                 │
                  │   CRITIC LLM   │
                  │   (gpt-4o-mini) │
                  │                 │
                  └────────┬────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
   ┌──────────────┐ ┌───────────┐ ┌──────────────┐
   │   TASK 1     │ │  TASK 2   │ │   TASK 3     │
   │              │ │           │ │              │
   │  Generate    │ │  Quality  │ │  Assign      │
   │  Ground      │ │  Check    │ │  Metadata    │
   │  Truth       │ │           │ │              │
   │  Answer      │ │  Pass/    │ │  Type,       │
   │              │ │  Fail     │ │  Difficulty   │
   └──────────────┘ └───────────┘ └──────────────┘
```

### Quality Filtering

```
   ┌──────────────────────────────────────────┐
   │          ❌ REJECTED                      │
   │                                          │
   │  • "What about Paul Graham?"             │
   │     → Too vague                          │
   │                                          │
   │  • "What's Graham's phone number?"       │
   │     → Not answerable from context        │
   │                                          │
   │  • Duplicate of existing question        │
   │     → Redundant                          │
   │                                          │
   │  • Broken grammar or unclear phrasing    │
   │     → Poorly formed                      │
   ├──────────────────────────────────────────┤
   │          ✅ ACCEPTED                      │
   │                                          │
   │  • Clear and specific                    │
   │  • Answerable from the provided context  │
   │  • Grammatically correct                 │
   │  • Unique — not a duplicate              │
   └──────────────────────────────────────────┘
```

> ⚠️ **This is why the critic model matters.** A weak model might accept vague questions or generate wrong answers. GPT-4o-mini is recommended because it follows structured output instructions reliably.

---

## 📊 Stage 5 — Final TestSet Output

> **Goal:** Compile all accepted Q&A pairs into a structured DataFrame.

```
testset.to_pandas()
```

| # | question | ground_truth | contexts | evolution_type |
|---|----------|-------------|----------|---------------|
| 1 | What did Paul Graham study at Cornell? | Philosophy | ["Paul Graham studied philosophy at Cornell before..."] | simple |
| 2 | How did Lisp influence Graham's startup philosophy? | It shaped his belief that programming languages influence thinking... | ["Graham's early experience with Lisp..."] | reasoning |
| 3 | Compare Graham's views on painting vs startup founders | Both painting and startups require creative vision and... | ["Graham saw painting as...", "At YC, Graham noticed..."] | multi_context |

**DataFrame Columns:**

| Column | Description |
|--------|-------------|
| `question` | The synthesized question |
| `ground_truth` | The correct answer derived from context |
| `contexts` | List of source chunks used |
| `evolution_type` | simple, reasoning, or multi_context |

---

## 🎯 How the TestSet Evaluates Your RAG

```
          ┌──────────────────────┐
          │   TestSet Questions  │
          │                      │
          │  Q1: "What did..."   │
          │  Q2: "How did..."    │
          │  Q3: "Compare..."    │
          └──────────┬───────────┘
                     │
            Feed questions to
                     │
                     ▼
          ┌──────────────────────┐
          │   YOUR RAG PIPELINE  │
          │                      │
          │  Retriever → LLM     │
          │                      │
          │  Returns: answers    │
          │  + retrieved contexts│
          └──────────┬───────────┘
                     │
               Compare with
               ground truth
                     │
                     ▼
          ┌──────────────────────┐
          │   RAGAS Evaluate     │
          └──────────┬───────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
┌─────────────┐ ┌──────────┐ ┌──────────────┐
│Faithfulness │ │ Answer   │ │ Context      │
│             │ │ Relevancy│ │ Precision    │
│ Is the      │ │          │ │              │
│ answer      │ │ Does it  │ │ Were the     │
│ grounded in │ │ actually │ │ right chunks │
│ retrieved   │ │ address  │ │ retrieved?   │
│ context?    │ │ the      │ │              │
│             │ │ question?│ │              │
│ Score: 0.85 │ │Score:0.92│ │ Score: 0.78  │
└─────────────┘ └──────────┘ └──────────────┘
```

**What Each Metric Tells You:**

| Metric | Score Range | What It Measures | Low Score Means |
|--------|-----------|-----------------|-----------------|
| **Faithfulness** | 0 → 1 | Is the answer supported by context? | LLM is hallucinating |
| **Answer Relevancy** | 0 → 1 | Does the answer address the question? | LLM is off-topic |
| **Context Precision** | 0 → 1 | Are retrieved chunks relevant? | Retriever needs tuning |
| **Context Recall** | 0 → 1 | Are all needed chunks retrieved? | Missing relevant docs |

---

## 🔄 API Changes: v0.1 vs v0.4

| Feature | ragas 0.1.x | ragas 0.4.x |
|---------|------------|------------|
| **Constructor** | `TestsetGenerator.from_langchain()` | `TestsetGenerator()` |
| **LLMs needed** | 2 (generator + critic) | 1 (single LLM) |
| **Wrappers** | Not needed | `LangchainLLMWrapper`, `LangchainEmbeddingsWrapper` |
| **Evolutions** | `simple, reasoning, multi_context` | Handled internally |
| **Param name** | `test_size` | `testset_size` |

**v0.1.x (old):**

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
)

testset = generator.generate_with_langchain_docs(
    documents,
    test_size=10,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)
```

**v0.4.x (new):**

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

generator = TestsetGenerator(
    llm=LangchainLLMWrapper(raw_llm),
    embedding_model=LangchainEmbeddingsWrapper(raw_emb),
)

testset = generator.generate_with_langchain_docs(
    documents,
    testset_size=10,
)
```

---

## 💻 Complete Working Code

```python
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

# ── API Key ──
OPENROUTER_KEY = "your-openrouter-key"

# ── Load & Chunk Documents ──
loader = DirectoryLoader(
    "./paul_graham",
    glob="**/*.*",
    loader_cls=TextLoader,
    loader_kwargs={"autodetect_encoding": True},
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
documents = loader.load_and_split(text_splitter)

for doc in documents:
    doc.metadata["filename"] = doc.metadata["source"]

print(f"Loaded {len(documents)} chunks")

# ── LLM & Embeddings (raw) ──
raw_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    temperature=0,
)

raw_emb = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_key=OPENROUTER_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
)

# ── Quick Smoke Test ──
print(raw_llm.invoke("Say hello").content)
r = raw_emb.embed_query("test")
print(f"Embedding dimension: {len(r)}")

# ── Wrap for Ragas 0.4.x ──
generator_llm = LangchainLLMWrapper(raw_llm)
embeddings = LangchainEmbeddingsWrapper(raw_emb)

# ── Generate TestSet ──
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=embeddings,
)

testset = generator.generate_with_langchain_docs(
    documents,
    testset_size=5,
)

# ── View Results ──
test_df = testset.to_pandas()
print(test_df.head())
```

---

## 🧭 Model Selection Guide

| Use Case | Recommended Model | Provider | Why |
|----------|------------------|----------|-----|
| **Generator + Critic** | `gpt-4o-mini` | OpenRouter | Reliable JSON, cheap, fast |
| **Embeddings** | `text-embedding-3-small` | OpenRouter | Good quality, low cost |
| **Fast prototyping** | `llama-3.3-70b` | Groq | Blazing fast, free tier |
| **Best quality** | `gpt-4o` | OpenRouter | Most accurate, expensive |

**Cost Comparison (approximate per 5 questions):**

| Setup | Cost | Speed |
|-------|------|-------|
| Groq (free) | $0.00 | ⚡ Fast but rate limited |
| OpenRouter GPT-4o-mini | ~$0.01 | 🚀 Fast |
| OpenRouter GPT-4o | ~$0.10 | 🐢 Slower |
| Local Ollama | $0.00 | 🐌 Slowest, JSON issues |

---

<p align="center">
  <sub>Built with 🧪 RAGAS · 🦜 LangChain · 🤖 OpenRouter</sub>
</p>