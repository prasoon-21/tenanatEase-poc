from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ExtractedLeaseSchema(BaseModel):
    """Strict Pydantic schema for Lease & KYC Document Extraction Pipeline."""
    lease_id: str = Field(..., description="Unique Lease Identifier")
    property_name: str = Field(..., description="Building or Property Name in TenantEase")
    unit_number: str = Field(..., description="Apartment/Commercial Unit Number")
    tenant_name: str = Field(..., description="Full legal name of tenant")
    landlord_name: str = Field(..., description="Full legal name of property owner/entity")
    monthly_rent_inr: float = Field(default=0.0, ge=0, description="Monthly base rent in INR (0.0 if unextracted)")
    security_deposit_inr: float = Field(default=0.0, ge=0, description="Security deposit in INR")
    due_day_of_month: int = Field(default=5, ge=1, le=28, description="Day of month rent is due")
    grace_period_days: int = Field(default=3, ge=0, le=15, description="Grace period before late fee applies")
    late_fee_inr_per_day: float = Field(default=0.0, ge=0, description="Late fee in INR per day after grace period")
    lease_start_date: str = Field(..., description="ISO date YYYY-MM-DD")
    lease_end_date: str = Field(..., description="ISO date YYYY-MM-DD")
    lock_in_months: int = Field(default=0, ge=0, description="Mandatory lock-in period in months")
    notice_period_days: int = Field(default=30, ge=0, description="Required notice period in days")
    annual_escalation_pct: float = Field(default=0.0, ge=0, le=25.0, description="Annual rent escalation percentage")
    maintenance_slas: dict = Field(default_factory=dict, description="SLA hours by urgency tier")
    kyc_id_type: str = Field(..., description="Verified ID document type (PAN / Aadhaar / Passport)")
    kyc_id_masked: str = Field(..., description="Masked KYC identifier for privacy compliance")
    extraction_confidence: float = Field(..., ge=0.0, le=1.0, description="Schema extraction confidence score")
    requires_hitl_review: bool = Field(..., description="True if confidence < 0.85 or anomaly detected")
    hitl_reasons: List[str] = Field(default_factory=list, description="Explanations for why HITL review is required")


class CitationChunk(BaseModel):
    chunk_id: str
    document_name: str
    clause_section: str
    page_number: int
    relevance_score: float
    verbatim_excerpt: str


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationChunk]
    groundedness_score: float
    latency_ms: float
    guardrail_status: Literal["PASSED", "BLOCKED_PROMPT_INJECTION", "FALLBACK_INSUFFICIENT_EVIDENCE"]


class TenantRiskProfile(BaseModel):
    tenant_id: str
    tenant_name: str
    unit_number: str
    monthly_rent_inr: float
    days_until_due: int
    past_6m_late_count: int
    avg_delay_days: float
    payment_bounce_count: int
    open_maintenance_tickets: int
    lease_remaining_months: int
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    top_risk_drivers: List[str]
    recommended_intervention: str
    suggested_nudge_channel: Literal["WhatsApp_Gentle", "SMS_Standard", "AI_Voice_Callback", "Human_Manager_Escalation"]
    requires_hitl_approval: bool


class MaintenanceTicketRequest(BaseModel):
    ticket_id: str
    unit_number: str
    tenant_name: str
    raw_description: str
    reported_timestamp: str


class MaintenanceTriageResponse(BaseModel):
    ticket_id: str
    unit_number: str
    category: Literal["PLUMBING", "ELECTRICAL", "HVAC", "STRUCTURAL_SAFETY", "APPLIANCE", "GENERAL"]
    urgency_tier: Literal["P0_EMERGENCY", "P1_HIGH", "P2_MEDIUM", "P3_LOW"]
    sla_target_hours: int
    estimated_cost_inr: float
    landlord_vs_tenant_liability: Literal["LANDLORD", "TENANT", "SHARED_REVIEW"]
    cited_lease_clause: str
    assigned_vendor_role: str
    hitl_sandbox_gate: bool = Field(..., description="True if estimated_cost_inr > 5000 or P0_EMERGENCY")
    hitl_reason: str
    automated_tenant_reply: str
