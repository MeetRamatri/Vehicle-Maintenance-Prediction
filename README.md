# 🚛 Vehicle Maintenance Prediction & Agentic Fleet Management System

An AI-driven analytics platform that predicts vehicle maintenance needs using historical sensor data and employs an Agentic AI assistant to generate structured servicing recommendations.

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

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* A free API Key from [Groq](https://console.groq.com/) or [Hugging Face](https://huggingface.co/settings/tokens)

---

## Milestones
### Milestone 1: ML-Based Prediction
* [ ] Data preprocessing and feature engineering.
* [ ] Implementation of Classification/Regression models.
* [ ] Feature importance analysis (SHAP/LIME).
* [ ] Basic UI for CSV uploads and risk visualization.
### Milestone 2: Agentic Assistant
* [ ] Integration of Open-source LLM via LangGraph.
* [ ] Implementation of State Management for the AI Agent.
* [ ] Automated generation of structured fleet reports.
* [ ] Extension: PDF Export of maintenance reports.

---

## Team Members
* Member 1: Meet Ramatri 
* Member 2: Anurag Singh Tomar
* Member 3: Aditya Prakash
* Member 4: Aditya Mohan Samadiya

