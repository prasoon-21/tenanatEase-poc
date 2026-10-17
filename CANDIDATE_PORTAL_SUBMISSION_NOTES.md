# Cynbit Candidate Portal — Copy-Paste Submission Guide & Technical Setup Notes

**Candidate:** Prasoon Kumar (`prasoonmishr.21@gmail.com` | `+91 7050514058`)  
**Role:** [AI/ML Engineer Intern](https://www.cynbit.in/careers/ai-ml-engineer-intern)  
**Dedicated Candidate Submission Portal:** [Submit Your Work — Cynbit Talent Hub](https://www.cynbit.in/candidate/cmuno8475000904jqp1ehxuz5)  
**Deadline:** Saturday, Oct 17, 2026 at 07:39 AM IST  

## 1. Quick 3-Command GitHub Push Instructions

Unzip `cynbit_tenantease_ai_poc_repo.zip` (which already contains a clean 6-commit Git history authored under your name and email) and push it to a new public repository on your [GitHub (@prasoon-21)](https://github.com/prasoon-21):

```bash
unzip cynbit_tenantease_ai_poc_repo.zip
cd cynbit-tenantease-ai-poc
git remote add origin https://github.com/prasoon-21/cynbit-tenantease-ai-poc.git
git push -u origin main
```

## 2. Copy-Paste "Technical Setup Notes" for the Cynbit Candidate Portal

Paste the text below directly into the **Technical Setup Notes** field on [cynbit.in/candidate/cmuno8475000904jqp1ehxuz5](https://www.cynbit.in/candidate/cmuno8475000904jqp1ehxuz5):

**Chosen AI Opportunity — Angle A: Evolving TenantEase into an AI-Native PropTech Co-Pilot**

After auditing Cynbit Technologies' public SaaS ecosystem at https://www.cynbit.in/products, I identified a high-ROI product asymmetry: while Cynbit Studio ("AI-Powered Career Growth Lab") and Revio ("AI-First Reputation Engine") already embed core AI workflows, TenantEase ("Smart Property & Tenant Management Platform") currently relies on rule-based rent reminders, manual lease lookup, unit vacancy tracking, and static KYC storage.

To evolve TenantEase into an AI-native platform while directly demonstrating all four core responsibilities of the Cynbit AI/ML Engineer Intern role (https://www.cynbit.in/careers/ai-ml-engineer-intern) and the Human-in-the-Loop (HITL) governance model from Cynbit's Oct 13, 2025 engineering post (https://www.cynbit.in/blog/ai-agents-business-operations), I built the **TenantEase AI Engine**:

1. **Lease & KYC Document Extraction Pipeline (`tenantease_ai/document_extractor.py`)**: Extracts 15 structured lease fields + masked Aadhaar/PAN KYC metadata from multi-page PDF agreements into validated Pydantic schemas (`ExtractedLeaseSchema`) with automated confidence & financial anomaly gating (`100.0%` schema pass rate across 3 multi-page Jaipur leases).
2. **Contextual Hybrid BM25 + TF-IDF Lease RAG Engine (`tenantease_ai/rag_engine.py`)**: Uses Document-Header Contextual Prepending + Hybrid Okapi BM25 & TF-IDF Cosine Similarity + Entity Reranking + a pre-retrieval Prompt-Injection Firewall. Upgrading from naive clause chunking (`60.0%` pass rate) to Contextual Retrieval achieved a **100.0% Golden QA Pass Rate**, **0.910 citation groundedness**, **1.59 ms average latency**, and a **100.0% block rate** on adversarial jailbreak/PII-leakage queries.
3. **Predictive Rent Delinquency Scorer (`tenantease_ai/risk_and_triage.py`)**: Replaces static calendar reminders with a calibrated logistic probability model combining payment delay history, UPI/NACH bounces, and open maintenance tickets. Evaluated across a 200-unit Jaipur portfolio (`INR 49.77 Lakhs/month` total rent), it achieves **82.5% accuracy** and a **0.752 F1 score**, identifying `INR 12.59 Lakhs` in High/Critical at-risk rent and routing tailored nudges across SMS, WhatsApp, Cynbit AI Voice Agents, or HITL Manager Escalation.
4. **Autonomous Maintenance Triage Agent with HITL Sandbox Gate (`tenantease_ai/risk_and_triage.py`)**: Classifies maintenance urgency (`P0_EMERGENCY` 4h SLA to `P3_LOW` 72h SLA), resolves Landlord vs. Tenant repair liability under Lease Clauses 5.1/5.2, and triggers a strict HITL approval gate whenever estimated repair costs exceed `INR 5,000` (Clause 5.3) or urgency is `P0_EMERGENCY`.
5. **Regress-Style CI/CD Evaluation Harness (`tenantease_ai/eval_harness.py`)**: Runs automated schema validation, RAG faithfulness, and red-team prompt-injection tests on every commit (`outputs/poc_benchmark_results.json`).

**Repository Deliverables Included:**
- **AI Opportunity Report:** `docs/AI_Opportunity_Report_Cynbit_Prasoon_Kumar.pdf` (and `docs/AI_Opportunity_Report.md`)
- **Technical Approach Report:** `docs/Technical_Approach_Report_Cynbit_Prasoon_Kumar.pdf` (and `docs/Technical_Approach_Report.md`)
- **Functional AI/ML Proof of Concept:** `api.py` (FastAPI REST microservice), `app.py` (Interactive Streamlit UI), `cli_demo.py` (Zero-config CLI benchmark runner), `tenantease_ai/` core package, and `tests/test_tenantease_ai.py` (5/5 unit tests passing).

**Reproduction & Setup Steps (Zero External API Keys Required for Local Benchmark):**
1. Clone repository & install dependencies:
   `pip install -r requirements.txt`
2. Run deterministic end-to-end CLI benchmark & unit test suite:
   `python3 cli_demo.py && python3 tests/test_tenantease_ai.py`
3. Launch FastAPI microservice (Swagger UI at `http://localhost:8000/docs`):
   `uvicorn api:app --reload --port 8000`
4. Launch interactive Streamlit dashboard (`http://localhost:8501`):
   `streamlit run app.py`
