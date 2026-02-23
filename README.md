# 🚛 Vehicle Maintenance Prediction & Agentic Fleet Management System

An AI-driven analytics platform that predicts vehicle maintenance needs using historical sensor data and employs an Agentic AI assistant to generate structured servicing recommendations.

---

## 💡 What This Project Does

This project is a **Vehicle Maintenance Prediction System** that helps fleet managers predict when vehicles need maintenance before breakdowns occur. It does this by:

1. **Analyzing vehicle data** (mileage, engine hours, fault codes, tire/brake/battery condition) using Machine Learning models (Random Forest, XGBoost) to calculate a **risk score** for each vehicle.
2. **Providing an AI chatbot** powered by LLMs (Groq/Llama 3) that answers maintenance questions, reasons about vehicle health, and generates structured service recommendations.
3. **Retrieving maintenance guidelines** via a RAG (Retrieval-Augmented Generation) pipeline backed by FAISS vector search, so recommendations are grounded in real servicing best practices.
4. **Presenting everything in a web dashboard** (Streamlit) where users can upload fleet CSV data, visualize risk distributions, and export PDF maintenance reports.

In short: upload your fleet data → get risk predictions → ask the AI chatbot for advice → download a maintenance report.

---

## 📌 Project Overview
This project addresses the challenges of proactive fleet management. By combining traditional **Machine Learning** for risk prediction with **Agentic AI (LLMs)** for reasoning, the system helps fleet managers move from reactive repairs to predictive maintenance, reducing downtime and operational costs.

### Key Features
* **Predictive Analytics:** Supervised ML models to forecast maintenance risks and time-to-failure.
* **Agentic Reasoning:** A LangGraph-powered agent that analyzes vehicle health and suggests actionable service plans.
* **RAG Integration:** Retrieves best practices for vehicle servicing to provide evidence-based recommendations.
* **Fleet Dashboard:** A Streamlit-based interface for data upload, visualization, and report generation.

---

## 🏗️ System Architecture

The system is divided into two main components: the ML Pipeline for quantitative risk assessment and the Agentic Workflow for qualitative recommendation generation.



### Workflow:
1.  **Data Ingestion:** CSV upload containing mileage, engine hours, and fault codes.
2.  **ML Engine:** Features are engineered and passed through a trained model (e.g., Random Forest or XGBoost) to calculate a "Risk Score."
3.  **Agentic Layer:** The Risk Score and vehicle history are passed to an LLM agent.
4.  **Reasoning & Retrieval:** The agent queries maintenance guidelines and generates a structured fleet report.
5.  **Output:** The user receives a health summary, service timeline, and a PDF export.

---

## 🛠️ Tech Stack

### Machine Learning & Data
* **Languages:** Python 3.x
* **Libraries:** `pandas`, `NumPy`, `scikit-learn`
* **Dataset:** [Vehicle Maintenance Data (Kaggle)](https://www.kaggle.com/datasets/chavindudulaj/vehicle-maintenance-data)

### Agentic AI (Milestone 2)
* **LLM Engine:** Groq Cloud API (Llama 3) or Hugging Face Inference API (Free Tiers)
* **Orchestration:** `LangGraph` or `LangChain`
* **Vector Store:** `ChromaDB` or `FAISS` (for RAG)

### Interface & Deployment
* **UI Framework:** `Streamlit`
* **Hosting:** `Hugging Face Spaces` or `Streamlit Community Cloud`

---

## 📂 Project Structure

```
Vehicle-Maintenance-Prediction/
├── app.py                        # Streamlit web dashboard (main UI)
├── main.py                       # FastAPI backend server (REST API)
├── MaintenancePridictiveModel.ipynb  # Jupyter notebook for ML model training
├── requirements.txt              # Full dependency list
├── requirements-ml.txt           # ML/LLM-only dependencies (for local/GPU use)
├── chatbot/
│   ├── vehicle_ai.py             # Conversational AI agent with memory
│   └── memory.py                 # Conversation memory management
├── rag_pipeline/
│   ├── retriever.py              # RAG retriever using FAISS vector search
│   ├── chunking.py               # Document chunking for vector storage
│   └── index.faiss               # Pre-built FAISS index
├── features/
│   └── vehicle_features.csv      # Engineered feature dataset
├── evaluation/
│   ├── llm_evaluation.py         # LLM output evaluation scripts
│   └── faithfulness_report.md    # Faithfulness evaluation report
├── fine_tuning/
│   ├── lora_setup.py             # LoRA adapter configuration
│   └── train_lora.py             # LoRA fine-tuning script
├── llm_data/
│   ├── enrich_text.py            # Text enrichment for training data
│   ├── generate_instructions.py  # Instruction dataset generation
│   ├── generate_text.py          # Text generation utilities
│   ├── instruction_dataset.jsonl # Generated instruction dataset
│   └── text_enriched.csv         # Enriched text data
└── static/                       # UI assets (images, icons)
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* A free API Key from [Groq](https://console.groq.com/) or [Hugging Face](https://huggingface.co/settings/tokens)
* Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  For ML/LLM-heavy workloads on a local GPU server, use `requirements-ml.txt` instead.

### Running the Streamlit Dashboard
To launch the web dashboard, run:

```bash
streamlit run app.py
```
The dashboard opens at [http://localhost:8501](http://localhost:8501) where you can upload fleet CSV data, view risk visualizations, chat with the AI assistant, and export reports.

### Running the Backend Server
To start the FastAPI application, run the following command from the project root:

```bash
uvicorn main:app --reload
```
Once the server is running, you can access the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).


---

## Milestones
### Milestone 1: ML-Based Prediction
* [x] Data preprocessing and feature engineering.
* [x] Implementation of Classification/Regression models.
* [ ] Feature importance analysis (SHAP/LIME).
* [x] Basic UI for CSV uploads and risk visualization.
### Milestone 2: Agentic Assistant
* [x] Integration of Open-source LLM via LangGraph.
* [x] Implementation of State Management for the AI Agent.
* [x] Automated generation of structured fleet reports.
* [ ] Extension: PDF Export of maintenance reports.

---

## Team Members
* Member 1: Meet Ramatri 
* Member 2: Aaditya Mohan Samadiya(team leader)
* Member 3: Anurag Singh Tomar
* Member 4: Aditya prakash
