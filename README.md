# ⚖️ Legal Document Assistant — Agentic AI Capstone 2026

An intelligent AI-powered assistant designed for paralegals and junior lawyers to navigate Indian law, built using **LangGraph**, **ChromaDB**, **Groq (Llama 3.3)**, and **Streamlit**. This is the final capstone project for the **Agentic AI Course 2026** by Dr. Kanthi Kiran Sirra.

---

## 🎯 Project Overview

| Field | Detail |
|---|---|
| **Domain** | Legal Document Assistant — Indian Law |
| **User** | Paralegals, junior lawyers, and law students |
| **Problem** | Legal professionals spend significant time searching dense documents for routine answers, while affordable legal guidance is often inaccessible to the public. |
| **Success** | Agent provides grounded legal explanations, remembers user context, and achieves a faithfulness score ≥ 0.7. |

---

## 🏗️ Architecture

The system utilizes an 8-node state machine to process queries:

```text
User Question
     ↓
[memory_node] → Append to history, sliding window, extract user name
     ↓
[router_node] → LLM decides: retrieve / tool / memory_only
     ↓
[retrieval_node]  /  [tool_node]  /  [skip_node]
     ↓
[answer_node] → Grounded answer from 12-document context
     ↓
[eval_node] → Faithfulness score (0.0–1.0) → Retry loop if < 0.7
     ↓
[save_node] → Append to history → END
```

**Technical Stack:**
- 🧠 **LLM** — `llama-3.3-70b-versatile` via Groq API
- 🔍 **Embeddings** — `all-MiniLM-L6-v2` via SentenceTransformers
- 🗄️ **Vector DB** — ChromaDB (in-memory)
- 🔗 **Orchestration** — LangGraph `StateGraph` with `MemorySaver`
- 🖥️ **UI** — Streamlit

---

## 📚 Knowledge Base — 12 Legal Topics

The assistant is grounded in 12 specialized documents covering 7 key legal domains:

| # | Topic | Domain |
|---|---|---|
| 1 | Essential Elements of a Valid Contract | Contract Law |
| 2 | Breach & Remedies | Contract Law |
| 3 | Vitiating Factors | Contract Law |
| 4 | Rights of an Arrested Person | Criminal Procedure |
| 5 | Trial Process | Criminal Procedure |
| 6 | Filing a Suit | Civil Procedure |
| 7 | Negligence & Duty of Care | Tort Law |
| 8 | Strict and Vicarious Liability | Tort Law |
| 9 | Fundamental Rights (Part III) | Constitutional Rights |
| 10 | Copyright and Trademark Law | Intellectual Property |
| 11 | Admissibility of Evidence | Evidence Law |
| 12 | Burden of Proof | Evidence Law |

---

## 📁 File Structure

```text
FinalProject/
├── agent.py                  # Core agent logic, nodes, and graph assembly
├── capstone_streamlit.py     # Streamlit web UI
└──  day13_capstone.ipynb      # Complete development notebook with outputs
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone [https://github.com/your-username/legal-document-assistant.git](https://github.com/your-username/legal-document-assistant.git)
cd legal-document-assistant
```

### 2. Create requirements.txt
Ensure you have a `requirements.txt` file with the following content:
```text
langgraph
langchain-groq
chromadb
sentence-transformers
streamlit
duckduckgo-search
langchain-community
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key
**Windows (Command Prompt):**
```cmd
set GROQ_API_KEY=your_groq_api_key_here
```
**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```
**Mac/Linux:**
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Running the App

### Quick smoke test (verify agent works)
```bash
python agent.py
```
Expected output:
```text
✅ Legal Assistant agent compiled successfully.
```

### Launch Streamlit UI
```bash
streamlit run capstone_streamlit.py
```
Then open **http://localhost:8501** in your browser.

---

## 💬 Features

- **Legal Q&A** — Answers grounded strictly in the 12-topic legal knowledge base.
- **Conversational Memory** — Persists user context (e.g., "Keshav") across turns using `thread_id`.
- **Self-reflection eval** — Every answer is scored for faithfulness; scores below 0.7 trigger an automatic retry.
- **Safety Refusal** — Correctly refuses medical or non-legal questions to prevent hallucination.
- **Tool Use** — Dynamically fetches real-time data or performs web searches when required.

---

## 📊 Evaluation Results (RAGAS)

| Metric | Score | Meaning |
|---|---|---|
| **Average Faithfulness** | 0.94 | ✅ PASS — High grounding in KB documents |
| **Min Score** | 0.90 | ✅ PASS |
| **Max Score** | 1.00 | ✅ PASS |

> Note: All evaluation pairs scored above the mandatory 0.7 threshold for the capstone.

---

## 🛡️ Red-Team Test Results

| Test | Expected Response | Result |
|---|---|---|
| Out-of-scope (Medical/GDP) | Refuse medical/non-legal questions | ✅ PASS |
| Prompt injection | Maintain system instructions & hold prompt | ✅ PASS |
| Memory persistence | Recall user name correctly turn-to-turn | ✅ PASS |

---

## 👨‍🏫 Submission Details

- **Student:** Keshav Gangal
- **Roll Number:** 23052726
- **Batch:** 2027_Agentic AI (OE)
- **Course:** Agentic AI  Course 2026
- **Trainer:** Dr. Kanthi Kiran Sirra

---
