"""
Document Extraction Pipeline for TenantEase AI.
Extracts structured lease terms, KYC metadata, and SLA rules from PDF lease agreements
using pypdf text extraction, robust deterministic parsing, optional LLM structured fallback,
strict Pydantic schema validation, and explainable Human-in-the-Loop (HITL) review gating.
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from .schemas import ExtractedLeaseSchema


class LeaseDocumentExtractor:
    """
    Production-style Lease & KYC Document Extraction Pipeline.
    Supports deterministic multi-pattern parsing from PDF text with optional LLM-assisted
    fallback extraction, strict Pydantic validation, and explainable HITL gating.
    """
    CONFIDENCE_THRESHOLD = 0.85

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            for idx, page in enumerate(reader.pages):
                pages.append({
                    "page_number": idx + 1,
                    "text": page.extract_text() or ""
                })
            return pages
        except Exception:
            return []

    def parse_lease_pdf(self, pdf_path: Path) -> ExtractedLeaseSchema:
        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            return ExtractedLeaseSchema(
                lease_id="TE-UNREADABLE-000",
                property_name="Unreadable Document",
                unit_number="U-000",
                tenant_name="Unknown",
                landlord_name="Unknown",
                monthly_rent_inr=0.0,
                security_deposit_inr=0.0,
                due_day_of_month=5,
                grace_period_days=3,
                late_fee_inr_per_day=0.0,
                lease_start_date="2026-01-01",
                lease_end_date="2026-12-31",
                lock_in_months=0,
                notice_period_days=30,
                annual_escalation_pct=0.0,
                kyc_id_type="Unknown",
                kyc_id_masked="XXXX",
                extraction_confidence=0.0,
                requires_hitl_review=True,
                hitl_reasons=["Corrupt, encrypted, empty, or unreadable PDF file."]
            )
        full_text = "\n".join(p["text"] for p in pages)
        return self.parse_lease_text(full_text, source_name=pdf_path.name)

    def _clean_num(self, val_str: str) -> float:
        if not val_str:
            return 0.0
        cleaned = re.sub(r"[^\d.]", "", val_str)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def parse_lease_text(self, text: str, source_name: str = "uploaded_lease.pdf") -> ExtractedLeaseSchema:
        def _find(patterns: List[str], default=None, cast=str):
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    try:
                        raw = m.group(1).strip()
                        if cast in (float, int):
                            return cast(self._clean_num(raw))
                        return cast(raw)
                    except Exception:
                        continue
            return default

        lease_id = _find([
            r"Lease ID:\s*([A-Z0-9\-]+)",
            r"Agreement (?:ID|No\.?):\s*([A-Z0-9\-]+)"
        ], "TE-UNKNOWN-000", str)

        property_name = _find([
            r"Property Name:\s*([^\n\r]+)",
            r"Premises:\s*([^\n\r]+)"
        ], "Unknown Property", str)

        unit_number = _find([
            r"Unit Number:\s*([A-Z0-9\-]+)",
            r"Flat No\.?:\s*([A-Z0-9\-]+)",
            r"Unit:\s*([A-Z0-9\-]+)"
        ], "U-000", str)

        tenant_name = _find([
            r"Tenant Name:\s*([^\n\r]+)",
            r"Tenant:\s*([^\n\r]+)"
        ], "Unknown Tenant", str)

        landlord_name = _find([
            r"Landlord Entity:\s*([^\n\r]+)",
            r"Landlord Name:\s*([^\n\r]+)",
            r"Lessor:\s*([^\n\r]+)"
        ], "Unknown Landlord", str)

        kyc_id_type = _find([
            r"KYC Verification Type:\s*([^\n\r]+)",
            r"ID Type:\s*([^\n\r]+)"
        ], "Unknown KYC", str)

        kyc_id_masked = _find([
            r"KYC Masked Reference:\s*([^\n\r]+)",
            r"Masked ID:\s*([^\n\r]+)"
        ], "XXXX", str)

        lease_start_date = _find([
            r"Lease Start Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            r"Commencement Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"
        ], "2026-01-01", str)

        lease_end_date = _find([
            r"Lease End Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
            r"Expiry Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"
        ], "2026-12-31", str)

        monthly_rent_inr = _find([
            r"(?:Monthly\s+Rent(?:\s+of)?|Rent Amount|Base Rent)[:\s]*(?:INR|Rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"pay a Monthly Rent of INR\s*([0-9,]+(?:\.[0-9]+)?)"
        ], 0.0, float)

        security_deposit_inr = _find([
            r"(?:Security\s+Deposit(?:\s+of)?|Deposit Amount)[:\s]*(?:INR|Rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"refundable Security Deposit of INR\s*([0-9,]+(?:\.[0-9]+)?)"
        ], 0.0, float)

        due_day_of_month = _find([
            r"on or before day\s*([0-9]+)\s*of each calendar month",
            r"Due Day(?:\s+of\s+Month)?:\s*([0-9]+)"
        ], 5, int)

        grace_period_days = _find([
            r"grace period of\s*([0-9]+)\s*calendar days",
            r"Grace Period:\s*([0-9]+)\s*days"
        ], 3, int)

        late_fee_inr_per_day = _find([
            r"Late Fee of INR\s*([0-9,]+(?:\.[0-9]+)?)\s*per day",
            r"Late Fee Penalty[:\s]*(?:INR|Rs\.?|₹)?\s*([0-9,]+(?:\.[0-9]+)?)"
        ], 0.0, float)

        annual_escalation_pct = _find([
            r"Annual Escalation of\s*([0-9]+(?:\.[0-9]+)?)%",
            r"Rent Escalation[:\s]*([0-9]+(?:\.[0-9]+)?)%"
        ], 0.0, float)

        lock_in_months = _find([
            r"Lock-In Period of\s*([0-9]+)\s*months",
            r"Lock-In:\s*([0-9]+)\s*months"
        ], 0, int)

        notice_period_days = _find([
            r"serving\s*([0-9]+)\s*days written notice",
            r"Notice Period:\s*([0-9]+)\s*days"
        ], 30, int)

        # Field completeness evaluation
        extracted_flags = [
            lease_id != "TE-UNKNOWN-000",
            property_name != "Unknown Property",
            unit_number != "U-000",
            tenant_name != "Unknown Tenant",
            monthly_rent_inr > 0,
            security_deposit_inr > 0,
            kyc_id_masked != "XXXX",
        ]
        confidence = round(sum(extracted_flags) / len(extracted_flags), 3)

        # Optional LLM structured extraction fallback if confidence is low and API key is set
        if confidence < self.CONFIDENCE_THRESHOLD and os.getenv("OPENAI_API_KEY"):
            llm_result = self._extract_with_llm(text)
            if llm_result:
                lease_id = llm_result.get("lease_id") or lease_id
                property_name = llm_result.get("property_name") or property_name
                unit_number = llm_result.get("unit_number") or unit_number
                tenant_name = llm_result.get("tenant_name") or tenant_name
                landlord_name = llm_result.get("landlord_name") or landlord_name
                if monthly_rent_inr <= 0 and llm_result.get("monthly_rent_inr"):
                    monthly_rent_inr = float(llm_result["monthly_rent_inr"])
                if security_deposit_inr <= 0 and llm_result.get("security_deposit_inr"):
                    security_deposit_inr = float(llm_result["security_deposit_inr"])
                if kyc_id_masked == "XXXX" and llm_result.get("kyc_id_masked"):
                    kyc_id_masked = str(llm_result["kyc_id_masked"])
                # Recalculate confidence
                extracted_flags = [
                    lease_id != "TE-UNKNOWN-000",
                    property_name != "Unknown Property",
                    unit_number != "U-000",
                    tenant_name != "Unknown Tenant",
                    monthly_rent_inr > 0,
                    security_deposit_inr > 0,
                    kyc_id_masked != "XXXX",
                ]
                confidence = round(sum(extracted_flags) / len(extracted_flags), 3)

        # Clamp numeric parameters to valid schema bounds to prevent crashes on malformed inputs
        due_day_of_month = min(28, max(1, due_day_of_month))
        grace_period_days = min(15, max(0, grace_period_days))
        late_fee_inr_per_day = max(0.0, late_fee_inr_per_day)
        annual_escalation_pct = min(25.0, max(0.0, annual_escalation_pct))
        lock_in_months = max(0, lock_in_months)
        notice_period_days = max(0, notice_period_days)

        # Explainable HITL gating reasons
        hitl_reasons = []
        if not text.strip():
            hitl_reasons.append("Empty or unextractable document text (scanned PDF or blank file).")
        if monthly_rent_inr <= 0:
            hitl_reasons.append("Monthly rent could not be extracted with verified confidence.")
        if security_deposit_inr > 0 and monthly_rent_inr > 0 and security_deposit_inr > (6 * monthly_rent_inr):
            hitl_reasons.append(
                f"Security deposit (INR {security_deposit_inr:,.0f}) exceeds 6x monthly rent (INR {monthly_rent_inr:,.0f})."
            )
        if annual_escalation_pct > 15.0:
            hitl_reasons.append(f"Annual rent escalation ({annual_escalation_pct}%) exceeds 15% standard cap.")
        if lease_end_date <= lease_start_date:
            hitl_reasons.append(f"Temporal anomaly: Lease end date ({lease_end_date}) is on or before start date ({lease_start_date}).")
        if confidence < self.CONFIDENCE_THRESHOLD:
            hitl_reasons.append(
                f"Extraction confidence score ({confidence:.2f}) is below minimum threshold ({self.CONFIDENCE_THRESHOLD})."
            )

        requires_hitl = len(hitl_reasons) > 0

        return ExtractedLeaseSchema(
            lease_id=lease_id,
            property_name=property_name,
            unit_number=unit_number,
            tenant_name=tenant_name,
            landlord_name=landlord_name,
            monthly_rent_inr=monthly_rent_inr,
            security_deposit_inr=security_deposit_inr,
            due_day_of_month=due_day_of_month,
            grace_period_days=grace_period_days,
            late_fee_inr_per_day=late_fee_inr_per_day,
            lease_start_date=lease_start_date,
            lease_end_date=lease_end_date,
            lock_in_months=lock_in_months,
            notice_period_days=notice_period_days,
            annual_escalation_pct=annual_escalation_pct,
            maintenance_slas={
                "P0_EMERGENCY": 4,
                "P1_HIGH": 24,
                "P2_MEDIUM": 48,
                "P3_LOW": 72,
            },
            kyc_id_type=kyc_id_type,
            kyc_id_masked=kyc_id_masked,
            extraction_confidence=confidence,
            requires_hitl_review=requires_hitl,
            hitl_reasons=hitl_reasons,
        )

    def _extract_with_llm(self, text: str) -> Optional[Dict[str, Any]]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = (
                "Extract structured lease parameters from this agreement text into valid JSON with these keys:\n"
                "- lease_id (string)\n"
                "- property_name (string)\n"
                "- unit_number (string)\n"
                "- tenant_name (string)\n"
                "- landlord_name (string)\n"
                "- monthly_rent_inr (number)\n"
                "- security_deposit_inr (number)\n"
                "- kyc_id_masked (string)\n\n"
                f"Agreement Text:\n{text[:3500]}\n\n"
                "Return ONLY valid JSON."
            )
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a legal document extraction specialist. Output valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return None
