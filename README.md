# Advanced AI Medical Intelligence Platform

An end-to-end medical vision platform designed to assist radiologists by evaluating chest radiographs, highlighting pathological structures using Explainable AI (XAI), and auto-drafting clinical reports.

## 🚀 Live Application URL
**Production Link:** [Insert Your Streamlit App Live Link Here]

## 🛠️ System Architecture & Engineering Breakdown
- **Deep Learning Engine (PyTorch):** Utilizes a pre-trained **DenseNet121** architecture optimized for medical imaging tasks. 
- **Explainable AI (XAI):** Implements specialized tensor-level backward gradient hooks to construct an isolated **Grad-CAM** localization heatmap overlay, bypassing traditional view mutation blocks.
- **Relational Data Persistence (SQLAlchemy):** Backed by an embedded **SQLite** storage tier running an automated schema initialization system for historical record logging (`PredictionHistory`).
- **Presentation Workspace (Streamlit):** Built a web dashboard handling real-time binary streaming, state updates, and interactive patient histories.

## 📁 Repository Directory Structure
```text
advanced-medical-ai/
├── core/
│   └── gradcam.py        # DL Core & Tensor-level Grad-CAM hooks
├── backend/
│   ├── database.py       # SQLite engine, ORM models, & dependency streams
│   └── llm_service.py    # Local automated clinical reporting blueprint
├── frontend/
│   └── app.py            # Streamlit multi-column interactive layout
├── requirements.txt      # Clean, standard Python library manifest
├── .python-version       # Fixed stable runtime instruction (Python 3.11)
└── README.md             # Engineering deployment documentation
```

## 💻 Local Installation & Setup Guide
1. Clone the project locally:
   ```bash
   git clone https://github.com
   cd Advanced-Medical-Ai
   ```
2. Initialize and activate a isolated python environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```
3. Install the dependencies and initiate local services:
   ```bash
   pip install -r requirements.txt
   streamlit run frontend/app.py
   ```