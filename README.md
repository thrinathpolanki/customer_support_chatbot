<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=200&section=header&text=Intelligent%20Customer%20Support%20Chatbot&fontSize=34&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Hybrid%20AI%20%7C%20Intent%20Classification%20%2B%20Generative%20Fallback&descAlignY=58&descSize=16" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=800&color=764ABA&center=true&vCenter=true&width=650&lines=Understands+customer+queries+with+AI;Hybrid+Tier-1+%2B+Tier-2+Architecture;Sentence-Transformers+%2B+FLAN-T5;Context-Aware+%7C+Fast+%7C+Reliable" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-FLAN--T5-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-informational?style=for-the-badge)

![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![Made With](https://img.shields.io/badge/Made%20with-❤️%20and%20Python-red?style=flat-square)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square)

</div>

<br/>

## 📌 About The Project

**Intelligent Customer Support Chatbot** is a hybrid conversational AI system built to answer customer support queries the way real production chatbots do — combining a **fast, deterministic intent classifier** for common questions with a **generative language model fallback** for anything open-ended.

Instead of relying on a single giant LLM for every message (slow, expensive, and prone to hallucinating policy details), this project uses a **two-tier architecture**:

```
                       ┌────────────────────────┐
        User Message ─▶│  Sentence-Transformer   │
                       │   (all-MiniLM-L6-v2)    │
                       └────────────┬────────────┘
                                    ▼
                       ┌────────────────────────┐
                       │  Logistic Regression +  │
                       │ Cosine-Similarity Boost │───▶ Confidence Score
                       └────────────┬────────────┘
                                    │
                     ┌──────────────┴───────────────┐
                     ▼                               ▼
          Confidence ≥ Threshold             Confidence < Threshold
                     │                               │
                     ▼                               ▼
        🎯 Curated, business-approved       🧠 FLAN-T5 generative
           response (instant, safe)            fallback (context-aware)
```

This means:
- ✅ Common questions (order status, refunds, shipping, returns...) get **instant, accurate, pre-approved answers**
- ✅ Open-ended or unexpected questions still get a **sensible, generated response** instead of a dead end
- ✅ The bot **remembers conversation context** across turns, so follow-up questions make sense
- ✅ Built-in **guardrails** detect and block degenerate/echoed generations before they ever reach a user

---

## ✨ Why This Project Is Useful

| Problem in typical chatbot demos | How this project solves it |
|---|---|
| Single LLM for everything → slow & unpredictable | Two-tier design: instant curated replies for known intents |
| No confidence signal → bot guesses blindly | Blended confidence = softmax probability **+** cosine similarity |
| Generative models hallucinate policy details | Fallback is only used for genuinely out-of-scope queries |
| No memory across turns | Per-session conversation context feeds the fallback model |
| Small models often "echo" prompts back | Built-in degeneration/echo detector with safe fallback message |
| Hard to demo to non-technical people | Polished Streamlit chat UI included, ready to present |

---

## 🧠 Key Features

- 🎯 **Semantic Intent Classification** — 11 pre-built customer support intents (order status, refunds, shipping, complaints, human escalation, etc.)
- 🧮 **Calibrated Confidence Scoring** — blends logistic regression probability with embedding cosine similarity for reliable thresholds
- 🧠 **Generative Fallback (FLAN-T5)** — handles anything outside known intents, using recent conversation history as context
- 🛡️ **Degeneration Guard** — automatically detects broken/echoed generations and swaps in a safe response
- 💭 **Multi-turn Context Memory** — per-session history so follow-up questions are understood correctly
- 🏷️ **Entity Extraction** — automatically pulls order IDs from free text and fills them into responses
- 🚀 **FastAPI REST Backend** — production-style API with interactive Swagger docs at `/docs`
- 🖥️ **Streamlit Chat UI** — live intent + confidence display for easy demoing
- 📊 **Automated Evaluation Pipeline** — accuracy, precision/recall/F1, and confusion matrix on a held-out test set

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| Language | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) Python 3.11+ |
| Embeddings | ![HuggingFace](https://img.shields.io/badge/sentence--transformers-FFD21E?style=flat-square) `all-MiniLM-L6-v2` |
| Classifier | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) Logistic Regression + cosine similarity |
| Generative Model | ![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-FLAN--T5-FFD21E?style=flat-square) `google/flan-t5-base` |
| Deep Learning Backend | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) |
| REST API | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) + Uvicorn |
| Demo UI | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) |
| Data Format | JSON (intents & test data) |

</div>

---

## 📂 Project Structure

```
customer_support_chatbot/
├── data/
│   ├── intents.json          # Training data: intents, patterns, responses
│   └── test_data.json        # Held-out labeled data for evaluation
├── models/                   # Generated by train.py (git-ignored)
│   ├── intent_classifier.joblib
│   ├── label_encoder.joblib
│   ├── intent_centroids.joblib
│   └── evaluation_report.txt
├── src/
│   ├── config.py              # Central configuration constants
│   ├── preprocessing.py       # Text cleaning + entity extraction
│   ├── intent_classifier.py   # Embedding + classification + confidence logic
│   ├── context_manager.py     # Per-session conversation memory
│   ├── response_generator.py  # Templated + generative response logic
│   ├── chatbot_engine.py      # Orchestrates the full pipeline
│   └── evaluate.py            # Model evaluation script
├── train.py                   # Trains and saves the intent classifier
├── api.py                     # FastAPI backend (production interface)
├── app.py                     # Streamlit UI (demo interface)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Getting Started — Run It On Your Own Machine

### ✅ Prerequisites

- **Python 3.11 or 3.12** (recommended — widest ML library wheel support)
- **pip** (comes with Python)
- ~2 GB free disk space (for model downloads on first run)

> 💡 On Windows, verify your Python version first:
> ```powershell
> python --version
> ```

### 📥 Step 1 — Clone / Pull the Repository

Open **PowerShell** and run:

```powershell
git clone https://github.com/thrinathpolanki/customer-support-chatbot.git
cd customer-support-chatbot
```

*(Or simply unzip the project folder and `cd` into it.)*

### 🐍 Step 2 — Create & Activate a Virtual Environment

```powershell
python -m venv venv
venv\Scripts\activate
```

You should now see `(venv)` at the start of your PowerShell prompt.

### 📦 Step 3 — Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 🧠 Step 4 — Train the Intent Classifier

```powershell
python train.py
```

This builds the classifier, label encoder, and intent centroids and saves them to `models/`.

### 📊 Step 5 — (Optional) Evaluate the Model

```powershell
python -m src.evaluate
```

Prints accuracy, per-intent precision/recall/F1, and a confusion matrix — and saves the report to `models/evaluation_report.txt`.

### 🚀 Step 6 — Run the App

**Option A — Streamlit Demo UI** (recommended for a live demo):

```powershell
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

**Option B — FastAPI REST Backend**:

```powershell
uvicorn api:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive Swagger API docs.

---

## 🧪 Testing It Out

Try these in the chat UI to see both tiers in action:

| Message | Expected Behavior |
|---|---|
| `"hi"` | 🎯 Greeting intent, high confidence |
| `"where is my order ORD12345"` | 🎯 Order status intent, order ID filled in |
| `"I want a refund"` | 🎯 Refund intent, high confidence |
| `"what's your favorite movie"` | 🧠 Generative fallback (out-of-scope) |
| `"I want to talk to a human"` | 🎯 Human agent intent + escalation flag |

Or test the API directly:

```powershell
curl -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d '{\"session_id\": \"test1\", \"message\": \"where is my order?\"}'
```

---

## 🩹 Troubleshooting

<details>
<summary><b>pip fails trying to build scikit-learn / numpy / torch from source</b></summary>
<br/>

This means your Python version doesn't have a prebuilt wheel for a pinned package. Use Python 3.11 or 3.12, or ensure `requirements.txt` uses minimum-version constraints (`>=`) rather than exact pins.
</details>

<details>
<summary><b>KeyError: "Unknown task text2text-generation"</b></summary>
<br/>

This project loads FLAN-T5 directly via `AutoModelForSeq2SeqLM` (not the `pipeline()` task-string API) specifically to avoid this issue across different `transformers` versions.
</details>

<details>
<summary><b>Bot uses the fallback for obvious messages like "hi"</b></summary>
<br/>

Re-run `python train.py` to rebuild `intent_centroids.joblib`, which powers the calibrated confidence boost. You can also lower `CONFIDENCE_THRESHOLD` in `src/config.py`.
</details>

<details>
<summary><b>Lots of "torchvision ModuleNotFoundError" warnings in the terminal</b></summary>
<br/>

These are harmless — Streamlit's file watcher probing unrelated vision-model files. They don't affect the chatbot. Install `torchvision` if you want to silence them (optional).
</details>

---

## 🗺️ Roadmap

- [ ] Fine-tune a transformer classifier head instead of logistic regression
- [ ] Add Retrieval-Augmented Generation (RAG) over a real knowledge base for the fallback tier
- [ ] Persist conversation context in Redis for multi-instance deployments
- [ ] Active-learning loop: log low-confidence queries for human review and periodic retraining
- [ ] Dockerize for one-command deployment

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check the [issues page](https://github.com/thrinathpolanki/customer-support-chatbot/issues) or open a pull request.

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

<div align="center">

## 👨‍💻 Author

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=20&duration=2500&pause=1000&color=667EEA&center=true&vCenter=true&width=400&lines=Polanki+Thrinath;AI%2FML+Enthusiast;Building+with+Python+%26+LLMs" alt="Author Typing SVG" />

### **Polanki Thrinath**

[![GitHub](https://img.shields.io/badge/GitHub-thrinathpolanki-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/thrinathpolanki)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-thrinathpolanki-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/thrinathpolanki)
[![Email](https://img.shields.io/badge/Email-polankithrinath%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:polankithrinath@gmail.com)

⭐ **If you found this project useful, consider giving it a star!** ⭐

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:764ba2,100:667eea&height=120&section=footer" width="100%"/>
