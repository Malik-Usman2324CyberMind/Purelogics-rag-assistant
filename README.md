# 🤖 PureLogics RAG Assistant

> A production-ready Retrieval-Augmented Generation (RAG) system that answers questions about **PureLogics** using only their public website data — with built-in **anti-hallucination guardrails**.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange)](https://groq.com)
[![sentence-transformers](https://img.shields.io/badge/Embeddings-sentence--transformers-yellow)](https://www.sbert.net)
[![Accuracy](https://img.shields.io/badge/Accuracy-88%25-brightgreen)]()

---
## 🎯 Problem Statement

Large Language Models (LLMs) like GPT-4, Claude, and LLaMA **hallucinate** when asked about specific companies or niche topics. They confidently generate plausible-sounding but **factually incorrect** answers.

**This project solves that problem** by combining:

1. **Retrieval** — Semantic search over verified PureLogics data
2. **Augmentation** — Injecting retrieved context into the prompt
3. **Generation** — Grounded LLM answer with source attribution
4. **Guardrails** — Refusing to answer when confidence is low

---
## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic Search** | Cosine similarity over `sentence-transformers` embeddings |
| 🎯 **Anti-Hallucination** | Similarity threshold + context-only prompting |
| 📚 **Source Attribution** | Every answer shows which file it came from |
| 🧪 **Evaluation Suite** | 25-question test set with hallucination checks |
| 🏗️ **Modular Architecture** | Clean `src/` structure (8 focused modules) |
| ⚡ **Fast Inference** | Groq's LPU inference — sub-second responses |
| 🐍 **Pure Python** | No heavy frameworks, easy to understand |

---
## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        USER QUERY                             │
│                  "When was PureLogics founded?"               │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  1. EMBED QUERY                                               │
│     sentence-transformers (all-MiniLM-L6-v2) → 384-dim vector │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  2. RETRIEVE (Cosine Similarity)                              │
│     Compare with 76 embedded chunks → Top-K = 3               │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  3. THRESHOLD CHECK (Anti-Hallucination Guardrail)            │
│     If best_score < 0.35 → "I don't know"                     │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  4. GENERATE (Groq LLM)                                       │
│     openai/gpt-oss-120b with context-only prompt              │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  5. ANSWER + SOURCE                                           │
│     "PureLogics was founded in 2006."  [Sources: about.txt]   │
└──────────────────────────────────────────────────────────────┘
```

---
## 📊 Evaluation Results

Tested on **25 hand-curated questions** across 6 categories:

| Category | Questions | Correct | Accuracy |
|----------|-----------|---------|----------|
| Company Basics | 4 | 4 | **100%** |
| Services & Expertise | 10 | 9 | **90%** |
| Industries | 2 | 2 | **100%** |
| Case Studies | 2 | 2 | **100%** |
| Careers | 2 | 2 | **100%** |
| **Hallucination Test** | 5 | 3 | **60%** |
| **TOTAL** | **25** | **22** | **88%** |

**Key Achievement:** System correctly refused to answer **3 out of 5** questions where information was not in the data — demonstrating robust anti-hallucination behavior.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Text → 384-dim vectors |
| **Vector Search** | NumPy cosine similarity | Fast retrieval |
| **LLM** | Groq `openai/gpt-oss-120b` | Grounded answer generation |
| **Data Processing** | Python 3.11, pandas | CSV handling |
| **Testing** | Custom evaluation framework | Quality assurance |

---
## 📁 Project Structure

```
Purelogics-rag-assistant/
│
├── README.md                    # You are here
├── LICENSE                      # MIT License
├── .gitignore                   # Python ignores
├── .env.example                 # API key template
├── requirements.txt             # Dependencies
├── main.py                      # Entry point
│
├── data/
│   ├── questions.csv            # 25-question evaluation set
│   ├── purelogics_qa_dataset.csv # 54 Q&A ground truth
│   └── raw/                     # 8 verified source files
│       ├── about.txt
│       ├── service_ai.txt
│       ├── service_mobile.txt
│       ├── service_web.txt
│       ├── service_qa.txt
│       ├── industries.txt
│       ├── case_study_1.txt
│       └── careers.txt
│
└── src/
    ├── __init__.py
    ├── config.py                # Central configuration
    ├── loader.py                # Loads .txt documents
    ├── chunker.py               # Splits into chunks
    ├── retriever.py             # Embedding + retrieval
    ├── generator.py             # LLM generation + guardrails
    ├── pipeline.py              # Orchestrates RAG flow
    └── evaluate.py              # Evaluation framework
```

---
## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Malik-Usman2324CyberMind/Purelogics-rag-assistant.git
cd Purelogics-rag-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# OR
.venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Key

Copy the example file and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 5. Run the Chatbot

```bash
python main.py
```

### 6. Run the Evaluation

```bash
python main.py eval
```

---

## 💬 Usage Example

### Interactive Mode

```
$ python main.py

Loading pipeline...
Pipeline ready with 76 chunks.

PureLogics RAG Chatbot (type 'exit' to quit)

You: When was PureLogics founded?

Bot: PureLogics was founded in 2006.
     Sources: about.txt

You: What is the CEO's salary?

Bot: I don't know based on the available information.

You: exit
```

### Programmatic Use

```python
from src.pipeline import RAGPipeline
import os

pipeline = RAGPipeline(os.getenv("GROQ_API_KEY"))

result = pipeline.ask("What AI services does PureLogics offer?")
print(result["answer"])
print(result["sources"])
```

---
## 📚 Data Sources

All data is collected from **publicly available** PureLogics sources and manually summarized in our own words:

- [PureLogics Company Page](https://purelogics.com/company/)
- [AI Services](https://purelogics.com/services/ai-api-development-and-integration/)
- [Mobile App Development](https://purelogics.com/services/mobile-app-development-services/)
- [Digital Commerce](https://purelogics.com/services/digital-commerce/)
- [Quality Assurance](https://purelogics.com/services/quality-assurance-services/)
- [Careers](https://purelogics.com/careers/)
- [ClosedWon Case Study](https://purelogics.com/resource_center/closedwon/)
- [LeadIQ Profile](https://leadiq.com/c/purelogics/5a1d98822300005900873397)

**Note:** We respect `robots.txt` and Terms of Use. All content is manually paraphrased — no scraping or copy-paste.

---

## 🎓 What I Learned

Building this project gave me hands-on experience with:

- **RAG Architecture** — Retrieval, augmentation, generation pipeline
- **Semantic Search** — Embeddings, cosine similarity, top-K retrieval
- **Anti-Hallucination Techniques** — Threshold-based refusal, context-only prompting
- **LLM APIs** — Groq's LPU inference with LLaMA models
- **Evaluation Frameworks** — Building test sets, measuring accuracy
- **Professional Python** — Modular code, type hints, docstrings
- **Git Workflow** — Version control, meaningful commits

---

## 🔮 Roadmap

- [x] Basic RAG pipeline
- [x] Anti-hallucination guardrails
- [x] Evaluation suite (25 questions)
- [ ] Add FAISS for faster retrieval at scale
- [ ] Streamlit web interface
- [ ] Multi-file source attribution
- [ ] Reranking with cross-encoders
- [ ] Docker deployment
- [ ] CI/CD with GitHub Actions

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Usman Malik**
BS Information Technology Graduate

- 📧 Email: [uzzimalik134@gmail.com](mailto:uzzimalik134@gmail.com)
- 💼 GitHub: [@Malik-Usman2324CyberMind](https://github.com/Malik-Usman2324CyberMind)

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

---

<p align="center">
  <b>Built with ❤️ using Python, Groq, and sentence-transformers</b>
</p>

