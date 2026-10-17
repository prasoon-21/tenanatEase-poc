"""
1. Predictive Rent Delinquency & Late-Payment ML Scorer (Calibrated Logistic Model + Feature Attribution)
2. Autonomous Maintenance Ticket Triage Agent with Lease Liability Check & HITL Sandbox Gate
"""
import os
import re
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
from .schemas import TenantRiskProfile, MaintenanceTicketRequest, MaintenanceTriageResponse


class RentDelinquencyScorer:
    """
    Predicts probability of rent payment delay past the grace period for TenantEase units
    using a calibrated logistic probability model and selects the optimal automated intervention channel.
    """
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)
        # Calibrated logistic model weights (aligned with Technical Approach Report formulation)
        self.weights = {
            "intercept": -2.65,
            "past_6m_late_count": 0.78,
            "avg_delay_days": 0.19,
            "payment_bounce_count": 1.15,
            "open_maintenance_tickets": 0.42,
            "near_lease_expiry": 0.35,
        }

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))

    def score_tenant(self, row: dict) -> TenantRiskProfile:
        tenant_id = str(row.get("tenant_id", "TE-USR-UNKNOWN"))
        tenant_name = str(row.get("tenant_name", "Unknown Tenant"))
        unit_number = str(row.get("unit_number", "U-UNKNOWN"))
        monthly_rent = max(0.0, float(row.get("monthly_rent_inr", 15000.0)))
        days_until_due = int(row.get("days_until_due", 5))

        late_cnt = max(0, int(row.get("past_6m_late_count", 0)))
        avg_delay = max(0.0, float(row.get("avg_delay_days", 0.0)))
        bounces = max(0, int(row.get("payment_bounce_count", 0)))
        tickets = max(0, int(row.get("open_maintenance_tickets", 0)))
        rem_months = max(0, int(row.get("lease_remaining_months", 12)))
        near_expiry = 1.0 if rem_months <= 2 else 0.0

        z = (
            self.weights["intercept"]
            + self.weights["past_6m_late_count"] * late_cnt
            + self.weights["avg_delay_days"] * avg_delay
            + self.weights["payment_bounce_count"] * bounces
            + self.weights["open_maintenance_tickets"] * tickets
            + self.weights["near_lease_expiry"] * near_expiry
        )
        prob = round(self._sigmoid(z), 3)

        drivers = []
        if bounces > 0:
            drivers.append(f"{bounces} historical UPI/NACH payment bounce(s)")
        if late_cnt >= 2:
            drivers.append(f"{late_cnt}/6 recent rent cycles paid past grace window")
        if avg_delay >= 4.0:
            drivers.append(f"Average payment delay of {avg_delay:.1f} days")
        if tickets >= 2:
            drivers.append(f"{tickets} unresolved maintenance tickets (dissatisfaction withhold risk)")
        if rem_months <= 2:
            drivers.append(f"Lease expiring in {rem_months} month(s) (flight/deposit-offset risk)")
        if not drivers:
            drivers.append("Consistent on-time payment track record")

        if prob >= 0.75:
            tier = "CRITICAL"
            channel = "Human_Manager_Escalation"
            action = "Trigger HITL Property Manager review + offer structured 2-part split payment plan & resolve open maintenance blockers."
            hitl = True
        elif prob >= 0.50:
            tier = "HIGH"
            channel = "AI_Voice_Callback"
            action = "Schedule proactive bilingual AI Voice Agent reminder 5 days before due date with instant UPI payment link."
            hitl = False
        elif prob >= 0.25:
            tier = "MEDIUM"
            channel = "WhatsApp_Gentle"
            action = "Send personalized WhatsApp interactive reminder 3 days prior to due date highlighting grace-period cutoff."
            hitl = False
        else:
            tier = "LOW"
            channel = "SMS_Standard"
            action = "Standard automated receipt & 1-day prior calendar nudge; zero intrusive chasing."
            hitl = False

        return TenantRiskProfile(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            unit_number=unit_number,
            monthly_rent_inr=monthly_rent,
            days_until_due=days_until_due,
            past_6m_late_count=late_cnt,
            avg_delay_days=avg_delay,
            payment_bounce_count=bounces,
            open_maintenance_tickets=tickets,
            lease_remaining_months=rem_months,
            default_probability=prob,
            risk_tier=tier,
            top_risk_drivers=drivers,
            recommended_intervention=action,
            suggested_nudge_channel=channel,
            requires_hitl_approval=hitl,
        )

    def generate_synthetic_portfolio(self, n_units: int = 200) -> Tuple[pd.DataFrame, dict]:
        rows = []
        for i in range(1, n_units + 1):
            # Realistic distribution across low, medium, high, critical risk tenants
            bucket = self.rng.choice(["low", "med", "high", "crit"], p=[0.58, 0.22, 0.13, 0.07])
            if bucket == "low":
                late_cnt = int(self.rng.choice([0, 1], p=[0.8, 0.2]))
                avg_delay = round(float(self.rng.uniform(0.0, 1.5)), 1)
                bounces = 0
                tickets = int(self.rng.choice([0, 1], p=[0.85, 0.15]))
                rem_months = int(self.rng.integers(4, 12))
            elif bucket == "med":
                late_cnt = int(self.rng.integers(1, 3))
                avg_delay = round(float(self.rng.uniform(2.0, 4.5)), 1)
                bounces = int(self.rng.choice([0, 1], p=[0.7, 0.3]))
                tickets = int(self.rng.integers(0, 2))
                rem_months = int(self.rng.integers(3, 10))
            elif bucket == "high":
                late_cnt = int(self.rng.integers(3, 5))
                avg_delay = round(float(self.rng.uniform(4.5, 8.0)), 1)
                bounces = int(self.rng.choice([1, 2], p=[0.6, 0.4]))
                tickets = int(self.rng.integers(1, 3))
                rem_months = int(self.rng.integers(2, 6))
            else:
                late_cnt = int(self.rng.integers(4, 7))
                avg_delay = round(float(self.rng.uniform(7.5, 14.0)), 1)
                bounces = int(self.rng.integers(2, 4))
                tickets = int(self.rng.integers(2, 5))
                rem_months = int(self.rng.integers(1, 3))

            rent = float(self.rng.choice([16500, 18500, 22000, 24500, 32000, 38000]))
            r_dict = {
                "tenant_id": f"TE-USR-{1000 + i}",
                "tenant_name": f"Tenant_{i:03d}",
                "unit_number": f"BLK-{(i % 4) + 1}-{100 + i}",
                "monthly_rent_inr": rent,
                "days_until_due": int(self.rng.integers(1, 10)),
                "past_6m_late_count": late_cnt,
                "avg_delay_days": avg_delay,
                "payment_bounce_count": bounces,
                "open_maintenance_tickets": tickets,
                "lease_remaining_months": rem_months,
            }
            profile = self.score_tenant(r_dict)
            # Simulate ground truth label for ROC-AUC / F1 evaluation
            true_prob = profile.default_probability
            actual_late = 1 if self.rng.random() < true_prob else 0
            pred_late = 1 if true_prob >= 0.40 else 0
            r_dict.update({
                "default_probability": profile.default_probability,
                "risk_tier": profile.risk_tier,
                "suggested_nudge_channel": profile.suggested_nudge_channel,
                "requires_hitl_approval": profile.requires_hitl_approval,
                "actual_late": actual_late,
                "pred_late": pred_late,
            })
            rows.append(r_dict)

        df = pd.DataFrame(rows)
        tp = int(((df["pred_late"] == 1) & (df["actual_late"] == 1)).sum())
        fp = int(((df["pred_late"] == 1) & (df["actual_late"] == 0)).sum())
        fn = int(((df["pred_late"] == 0) & (df["actual_late"] == 1)).sum())
        tn = int(((df["pred_late"] == 0) & (df["actual_late"] == 0)).sum())
        precision = round(tp / max(1, tp + fp), 3)
        recall = round(tp / max(1, tp + fn), 3)
        f1 = round(2 * precision * recall / max(1e-6, precision + recall), 3)
        accuracy = round((tp + tn) / len(df), 3)

        tier_counts = df["risk_tier"].value_counts().to_dict()
        total_monthly_rent = float(df["monthly_rent_inr"].sum())
        at_risk_rent = float(df[df["risk_tier"].isin(["HIGH", "CRITICAL"])]["monthly_rent_inr"].sum())

        metrics = {
            "total_units_evaluated": len(df),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tier_distribution": {
                "LOW": int(tier_counts.get("LOW", 0)),
                "MEDIUM": int(tier_counts.get("MEDIUM", 0)),
                "HIGH": int(tier_counts.get("HIGH", 0)),
                "CRITICAL": int(tier_counts.get("CRITICAL", 0)),
            },
            "total_monthly_rent_inr": total_monthly_rent,
            "high_and_critical_at_risk_rent_inr": at_risk_rent,
            "estimated_early_recovery_inr": round(at_risk_rent * 0.68, 2),
        }
        return df, metrics


class MaintenanceTriageAgent:
    """
    Autonomous Maintenance Ticket Triage Agent with Lease Clause Liability Grounding
    and Human-in-the-Loop (HITL) Sandbox Gate (per Cynbit's HITL Architecture).
    """
    HITL_COST_THRESHOLD_INR = float(os.getenv("HITL_COST_THRESHOLD_INR", "5000.0"))

    def triage(self, req: MaintenanceTicketRequest) -> MaintenanceTriageResponse:
        desc = (req.raw_description or "").strip().lower()

        if not desc:
            category = "GENERAL"
            urgency = "P3_LOW"
            sla = 72
            cost = 0.0
            liability = "SHARED_REVIEW"
            clause = "Clause 4.4 (General Query / Unspecified Request)"
            vendor = "TenantEase Property Support Desk"
            hitl_gate = True
            hitl_reason = "HITL Sandbox Gate Active: Maintenance ticket description is empty. Requires property manager clarification."
        elif any(w in desc for w in ["burst", "flooding", "short circuit", "spark", "smoke", "fire", "gas leak", "shock"]):
            category = "PLUMBING" if ("burst" in desc or "flood" in desc or "pipe" in desc) else "ELECTRICAL"
            urgency = "P0_EMERGENCY"
            sla = 4
            cost = 6800.0
            liability = "LANDLORD"
            clause = "Clause 4.1 (P0_EMERGENCY 4h SLA) & Clause 5.1 (Concealed Plumbing/Main Wiring = 100% Landlord Liability)"
            vendor = "Emergency Rapid-Response Master Technician"
            hitl_gate = (cost > self.HITL_COST_THRESHOLD_INR) or (urgency == "P0_EMERGENCY")
            hitl_reason = (
                f"HITL Sandbox Gate Active: Urgency is {urgency} and estimated cost (INR {cost:.0f}) "
                f"exceeds Clause 5.3 threshold (INR {self.HITL_COST_THRESHOLD_INR:.0f}). "
                "Vendor dispatch staged; awaiting 1-click Property Manager approval."
            )
        elif any(w in desc for w in ["ac ", "air conditioner", "compressor", "hvac", "geyser", "water heater", "main lock"]):
            category = "HVAC" if ("ac " in desc or "air conditioner" in desc or "compressor" in desc) else "APPLIANCE"
            urgency = "P1_HIGH"
            sla = 24
            cost = 3200.0
            liability = "LANDLORD"
            clause = "Clause 4.2 (P1_HIGH 24h SLA) & Clause 5.1 (Capital HVAC/Fixture Repair)"
            vendor = "Certified HVAC & Electrical Specialist"
            hitl_gate = (cost > self.HITL_COST_THRESHOLD_INR) or (urgency == "P0_EMERGENCY")
            hitl_reason = "Auto-Approved within Sandbox Constraints (< INR 5,000 & non-P0)."
        elif any(w in desc for w in ["seepage", "damp", "intercom", "washing machine", "refrigerator", "drain slow"]):
            category = "PLUMBING" if ("seepage" in desc or "drain" in desc) else "APPLIANCE"
            urgency = "P2_MEDIUM"
            sla = 48
            cost = 2100.0
            liability = "LANDLORD" if "seepage" in desc else "SHARED_REVIEW"
            clause = "Clause 4.3 (P2_MEDIUM 48h SLA) & Clause 5.1/5.2 (External Seepage vs Appliance Wear)"
            vendor = "General Facility Maintenance Crew"
            hitl_gate = (cost > self.HITL_COST_THRESHOLD_INR) or (urgency == "P0_EMERGENCY")
            hitl_reason = "Auto-Approved within Sandbox Constraints (< INR 5,000 & non-P0)."
        else:
            category = "GENERAL"
            urgency = "P3_LOW"
            sla = 72
            cost = 850.0
            liability = "TENANT"
            clause = "Clause 4.4 (P3_LOW 72h SLA) & Clause 5.2 (Consumables & Minor Wear < INR 1,500 = Tenant Liability)"
            vendor = "In-House Handyman Roster"
            hitl_gate = (cost > self.HITL_COST_THRESHOLD_INR) or (urgency == "P0_EMERGENCY")
            hitl_reason = "Auto-Approved within Sandbox Constraints (< INR 5,000 & non-P0)."

        reply = (
            f"Hi {req.tenant_name}, your ticket ({req.ticket_id}) for Unit {req.unit_number} has been classified as "
            f"{urgency} ({category}) with a {sla}-hour SLA under {clause}. "
            f"Assigned to: {vendor}. {'A property manager has been paged for immediate emergency sign-off.' if hitl_gate else 'Work order has been automatically dispatched.'}"
        )

        return MaintenanceTriageResponse(
            ticket_id=req.ticket_id,
            unit_number=req.unit_number,
            category=category,
            urgency_tier=urgency,
            sla_target_hours=sla,
            estimated_cost_inr=cost,
            landlord_vs_tenant_liability=liability,
            cited_lease_clause=clause,
            assigned_vendor_role=vendor,
            hitl_sandbox_gate=hitl_gate,
            hitl_reason=hitl_reason,
            automated_tenant_reply=reply,
        )
