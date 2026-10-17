"""
FastAPI Microservice for TenantEase AI Co-Pilot (Cynbit Technologies Assessment PoC).
Exposes REST API endpoints for:
- GET  / (and GET /health)
- POST /api/v1/extract-lease
- POST /api/v1/rag-query
- POST /api/v1/predict-rent-risk
- POST /api/v1/triage-maintenance
- GET  /api/v1/eval-suite
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment configuration (.env)
load_dotenv()

from tenantease_ai.pdf_generator import generate_sample_lease_pdfs
from tenantease_ai.document_extractor import LeaseDocumentExtractor
from tenantease_ai.rag_engine import HybridLeaseRAGEngine
from tenantease_ai.risk_and_triage import RentDelinquencyScorer, MaintenanceTriageAgent
from tenantease_ai.eval_harness import TenantEaseEvalHarness
from tenantease_ai.schemas import (
    ExtractedLeaseSchema,
    RAGQueryResponse,
    TenantRiskProfile,
    MaintenanceTicketRequest,
    MaintenanceTriageResponse,
)

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data" / "sample_leases"))
generate_sample_lease_pdfs(DATA_DIR)

extractor = LeaseDocumentExtractor()
rag_engine = HybridLeaseRAGEngine()
rag_engine.ingest_pdfs(sorted(DATA_DIR.glob("*.pdf")))
risk_scorer = RentDelinquencyScorer()
triage_agent = MaintenanceTriageAgent()
eval_harness = TenantEaseEvalHarness(DATA_DIR)

app = FastAPI(
    title="TenantEase AI Engine API — Cynbit Technologies PoC",
    description="AI Document Extraction, Citation-Grounded Lease RAG, Predictive Rent Risk & HITL Maintenance Triage",
    version="1.0.0",
)

# CORS configuration for frontend and client dashboards
cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
cors_origins = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LeaseTextPayload(BaseModel):
    text: str = Field(..., description="Raw text of the residential/commercial lease agreement")
    source_name: str = Field(default="uploaded_lease.pdf", description="Original file name or reference")


class RAGRequestPayload(BaseModel):
    query: str = Field(..., description="Lease clause, SLA, or policy question")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of citation chunks to retrieve")


class RiskInputPayload(BaseModel):
    tenant_id: str = "TE-USR-1042"
    tenant_name: str = "Aarav Sharma"
    unit_number: str = "A-402"
    monthly_rent_inr: float = 24500.0
    days_until_due: int = 4
    past_6m_late_count: int = 3
    avg_delay_days: float = 5.5
    payment_bounce_count: int = 1
    open_maintenance_tickets: int = 2
    lease_remaining_months: int = 2


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "service": "TenantEase AI Engine (Cynbit Technologies Assessment PoC)",
        "author": "Prasoon Kumar",
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "llm_generative_mode_active": bool(os.getenv("OPENAI_API_KEY")),
        "indexed_lease_chunks": len(rag_engine.chunks),
        "hitl_cost_threshold_inr": triage_agent.HITL_COST_THRESHOLD_INR,
        "endpoints": [
            "/health",
            "/api/v1/extract-lease",
            "/api/v1/rag-query",
            "/api/v1/predict-rent-risk",
            "/api/v1/triage-maintenance",
            "/api/v1/eval-suite",
        ],
    }


@app.post("/api/v1/extract-lease", response_model=ExtractedLeaseSchema)
def extract_lease_endpoint(payload: LeaseTextPayload):
    return extractor.parse_lease_text(payload.text, source_name=payload.source_name)


@app.post("/api/v1/rag-query", response_model=RAGQueryResponse)
def rag_query_endpoint(payload: RAGRequestPayload):
    return rag_engine.query(payload.query, top_k=payload.top_k)


@app.post("/api/v1/predict-rent-risk", response_model=TenantRiskProfile)
def predict_rent_risk_endpoint(payload: RiskInputPayload):
    return risk_scorer.score_tenant(payload.model_dump())


@app.post("/api/v1/triage-maintenance", response_model=MaintenanceTriageResponse)
def triage_maintenance_endpoint(payload: MaintenanceTicketRequest):
    return triage_agent.triage(payload)


@app.get("/api/v1/eval-suite")
def run_eval_suite_endpoint():
    return eval_harness.run_full_suite()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("api:app", host=host, port=port, reload=True)
