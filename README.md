# SafeOrch - Intelligent Safety Orchestration Platform

Batch-process safety footage through a multi-agent pipeline that detects PPE violations, scores risk contextually using OSHA standards, decides appropriate action, and generates explainable reports.

## Tech Stack

| Component               | Tool                          |
| ----------------------- | ----------------------------- |
| **CV Detection**        | YOLOv8 (fine-tuned on PPE)    |
| **Agent Orchestration** | LangGraph                     |
| **RAG**                 | LangChain + ChromaDB          |
| **LLM**                 | Groq Llama-3.3-70B / Nemotron |
| **Predictive Layer**    | Scikit-learn / XGBoost        |
| **Frontend**            | Streamlit                     |
| **Containerization**    | Docker                        |

---

🚧 **Work in progress** 🚧
