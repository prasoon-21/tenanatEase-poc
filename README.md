# TenantEase AI Engine — Cognitive PropTech Co-Pilot for Cynbit Technologies

**Target Product:** [TenantEase (Cynbit Proprietary SaaS Suite)](https://www.cynbit.in/products)  
**Role:** [AI/ML Engineer Intern — Cynbit Technologies](https://www.cynbit.in/careers/ai-ml-engineer-intern)  

---

## 1. Overview & Project Working

TenantEase is a smart property & tenant management platform designed to automate rent reminders, lease audits, unit vacancy tracking, and digital KYC storage. **TenantEase AI Engine** transforms this workflow into an AI-native co-pilot consisting of four modular pipelines governed by Human-in-the-Loop (HITL) sandbox controls:

1. **Lease & KYC Document Extraction Pipeline (`tenantease_ai/document_extractor.py`)**:
   - Parses multi-page PDF lease agreements (`pypdf`).
   - Extracts 15 structured parameters (Rent, Security Deposit, Due Day, Grace Period, Late Fee, Dates, Notice, Lock-in, Escalation, SLAs, masked Aadhaar/PAN) into a strict Pydantic contract (`ExtractedLeaseSchema`).
   - Dynamically flags unextracted fields and financial anomalies (e.g., Deposit > 6x Rent, Escalation > 15%) for Human-in-the-Loop review with explicit explanations.
   - Includes optional LLM structured fallback when an OpenAI API key is supplied.

2. **Contextual Hybrid Lease RAG Engine (`tenantease_ai/rag_engine.py`)**:
   - **Contextual Prepending**: Prepends document-level metadata (Tenant, Unit, Property, Lease ID) to each clause chunk to prevent cross-tenant clause ambiguity.
   - **Hybrid Retrieval**: Computes Okapi BM25 lexical scoring + dense TF-IDF cosine similarity + entity match boosting.
   - **Guardrails**: Pre-retrieval firewall immediately blocks prompt injections, jailbreaks, and PII exfiltration queries.
   - **Dual Generation**: Runs in fast, zero-token local extractive mode out-of-the-box, and automatically activates LLM generative synthesis with master system prompt when `OPENAI_API_KEY` is provided.

3. **Predictive Rent Delinquency Scorer (`tenantease_ai/risk_and_triage.py`)**:
   - Calibrated logistic probability model based on behavioral payment parameters (past 6-month late payment frequency, average delay days, UPI/NACH bounce history, open maintenance tickets, and near lease expiry).
   - Classifies units into `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` risk tiers.
   - Routes actionable nudges across standard SMS, interactive WhatsApp, AI Voice Agent callbacks, or HITL Property Manager escalation.

4. **Autonomous Maintenance Triage Agent (`tenantease_ai/risk_and_triage.py`)**:
   - Classifies tickets into categories (`PLUMBING`, `ELECTRICAL`, `HVAC`, `APPLIANCE`, `GENERAL`) and urgency tiers (`P0_EMERGENCY` 4h SLA to `P3_LOW` 72h SLA).
   - Maps Landlord vs. Tenant liability based on lease covenants (Clauses 5.1/5.2).
   - Automatically halts autonomous dispatch and activates a HITL approval sandbox if estimated repair cost exceeds `INR 5,000` (Clause 5.3) or urgency is `P0_EMERGENCY`.

5. **CI/CD Evaluation & Red-Teaming Harness (`tenantease_ai/eval_harness.py`)**:
   - Automates validation of schema parsing, golden RAG queries, prompt injection defenses, and triage governance rules.

---

## 2. Verified Benchmark Metrics

| Metric | Measured Value | Validation Source |
| :--- | :--- | :--- |
| **Unit Test Suite** | **7 / 7 Passing (100%)** | `tests/test_tenantease_ai.py` |
| **Document Extraction Pass Rate** | **100.0%** | `outputs/poc_benchmark_results.json` |
| **Golden RAG Pass Rate** | **100.0%** | 5 multi-clause golden questions |
| **RAG Lexical Groundedness** | **1.00** | Token overlap against verified citations |
| **RAG Retrieval Latency** | **~1.9 ms** | Local hybrid BM25 + TF-IDF index |
| **Prompt Injection Defense** | **100.0% Block Rate** | Red-teaming cases (`RED-01` to `RED-03`) |
| **Delinquency Classifier (200 units)** | **82.5% Accuracy, 0.752 F1** | `outputs/tenantease_200_units_risk_scored.csv` |
| **Maintenance HITL Compliance** | **100.0%** | Strict gating on cost > 5,000 INR & P0 |

---

## 3. Repository Structure

```text
cynbit-tenantease-ai-poc/
├── .env.example                           # Documented template for all environment variables
├── .env                                   # Local configuration file
├── api.py                                 # FastAPI REST microservice (/health, /api/v1/*)
├── app.py                                 # Interactive Streamlit operator dashboard
├── cli_demo.py                            # Zero-config CLI benchmark & report runner
├── Dockerfile                             # Containerized production deployment
├── entrypoint.sh                          # Container startup router (API vs Dashboard)
├── requirements.txt                       # Python dependencies
├── tenantease_ai/
│   ├── __init__.py
│   ├── schemas.py                         # Pydantic v2 data models & validation contracts
│   ├── pdf_generator.py                   # ReportLab multi-clause Jaipur lease PDF generator
│   ├── document_extractor.py              # Lease & KYC extraction pipeline with explainable HITL
│   ├── rag_engine.py                      # Contextual Hybrid BM25 + TF-IDF RAG with guardrails
│   ├── risk_and_triage.py                 # Rent Delinquency Scorer & HITL Maintenance Triage Agent
│   └── eval_harness.py                    # Automated CI/CD evaluation and adversarial red-teaming
├── data/sample_leases/                    # Generated Jaipur residential & commercial lease PDFs
├── outputs/
│   ├── poc_benchmark_results.json         # Complete verified benchmark & evaluation metrics
│   └── tenantease_200_units_risk_scored.csv # 200-unit portfolio risk predictions & nudge routing
├── docs/
│   ├── AI_Opportunity_Report.md           # Strategic AI Opportunity Report
│   └── Technical_Approach_Report.md       # Architecture & mathematical formulation report
└── tests/
    └── test_tenantease_ai.py              # Automated test suite (7/7 tests passing)
```

---

## 4. Environment Variables Reference

Copy `.env.example` to `.env` to configure the application:

| Variable | Default | Purpose / Notes |
| :--- | :--- | :--- |
| `PORT` | `8000` | Port for web server. Injected automatically by Cloud Run, Render, Railway. |
| `HOST` | `0.0.0.0` | Bind IP address for server listeners. |
| `ENVIRONMENT` | `production` | Deployment environment (`development`, `staging`, `production`). |
| `LOG_LEVEL` | `info` | Application logging verbosity (`debug`, `info`, `warning`, `error`). |
| `CORS_ORIGINS` | `*` | Allowed CORS origins for frontend integration (e.g. `https://tenantease.cynbit.in`). |
| `APP_MODE` | `api` | Container mode: `api` (FastAPI) or `dashboard` (Streamlit). |
| `OPENAI_API_KEY` | *(optional)* | OpenAI API key. If unset, engine runs in local deterministic mode. |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for generative answers and extraction fallback. |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature (0.0 enforces strict adherence to citations). |
| `HITL_COST_THRESHOLD_INR` | `5000.0` | Maintenance expense limit in INR before requiring human approval. |
| `EXTRACTION_CONFIDENCE_THRESHOLD` | `0.85` | Confidence floor before flagging lease for manual review. |
| `DATA_DIR` | `./data/sample_leases` | Directory where PDF leases are stored and indexed. |
| `OUTPUTS_DIR` | `./outputs` | Directory for benchmark JSON and portfolio risk CSV exports. |
| `STREAMLIT_SERVER_PORT` | `8501` | Listening port for Streamlit operator console. |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Headless execution flag for Streamlit in container environments. |

---

## 5. Local Setup & Quickstart

### Prerequisites
- Python 3.11 or 3.12
- Git

### Step 1: Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Run Verification Test Suite
```bash
python tests/test_tenantease_ai.py
```

### Step 3: Run Deterministic CLI Benchmark Runner
```bash
python cli_demo.py
```

### Step 4: Launch Services

**Option A: Launch FastAPI Microservice (Swagger UI at `http://localhost:8000/docs`)**
```bash
uvicorn api:app --reload --port 8000
```

**Option B: Launch Streamlit Operator Dashboard (`http://localhost:8501`)**
```bash
streamlit run app.py
```

---

## 6. Docker & Container Deployment

### Build Docker Image
```bash
docker build -t tenantease-ai:latest .
```

### Run FastAPI Microservice
```bash
docker run -p 8000:8000 -e APP_MODE=api tenantease-ai:latest
```

### Run Streamlit Operator Dashboard
```bash
docker run -p 8501:8501 -e APP_MODE=dashboard -e PORT=8501 tenantease-ai:latest
```
