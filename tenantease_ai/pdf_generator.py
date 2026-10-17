"""
Generates realistic multi-clause Indian Residential & Commercial Lease Agreement PDFs
for testing the TenantEase AI Document Extraction & Citation-Grounded RAG Pipelines.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


SAMPLE_LEASES_METADATA = [
    {
        "filename": "Lease_TE_JPR_101_Aarav_Sharma.pdf",
        "lease_id": "TE-JPR-2026-101",
        "property_name": "Cynbit Pratap Enclave, Pratap Nagar, Jaipur",
        "unit_number": "A-402",
        "tenant_name": "Aarav Sharma",
        "landlord_name": "Pratap Infra Estates Pvt. Ltd.",
        "monthly_rent_inr": 24500.0,
        "security_deposit_inr": 49000.0,
        "due_day_of_month": 5,
        "grace_period_days": 3,
        "late_fee_inr_per_day": 250.0,
        "lease_start_date": "2026-04-01",
        "lease_end_date": "2027-03-31",
        "lock_in_months": 6,
        "notice_period_days": 60,
        "annual_escalation_pct": 7.5,
        "kyc_id_type": "Aadhaar + PAN",
        "kyc_id_masked": "XXXX-XXXX-4821 / ABCPS****K",
    },
    {
        "filename": "Lease_TE_JPR_102_Neha_Verma.pdf",
        "lease_id": "TE-JPR-2026-102",
        "property_name": "Sitapura Tech Suites, Sitapura Industrial Area, Jaipur",
        "unit_number": "B-210",
        "tenant_name": "Neha Verma",
        "landlord_name": "Rajputana Commercial Holdings LLP",
        "monthly_rent_inr": 38000.0,
        "security_deposit_inr": 114000.0,
        "due_day_of_month": 7,
        "grace_period_days": 5,
        "late_fee_inr_per_day": 500.0,
        "lease_start_date": "2026-01-01",
        "lease_end_date": "2026-12-31",
        "lock_in_months": 9,
        "notice_period_days": 90,
        "annual_escalation_pct": 10.0,
        "kyc_id_type": "PAN",
        "kyc_id_masked": "BGLPV****M",
    },
    {
        "filename": "Lease_TE_JPR_103_Vikram_Rathore.pdf",
        "lease_id": "TE-JPR-2026-103",
        "property_name": "Malviya Residency Hub, Malviya Nagar, Jaipur",
        "unit_number": "C-105",
        "tenant_name": "Vikram Singh Rathore",
        "landlord_name": "PinkCity Smart Living Pvt. Ltd.",
        "monthly_rent_inr": 18500.0,
        "security_deposit_inr": 37000.0,
        "due_day_of_month": 5,
        "grace_period_days": 2,
        "late_fee_inr_per_day": 150.0,
        "lease_start_date": "2026-06-01",
        "lease_end_date": "2027-05-31",
        "lock_in_months": 6,
        "notice_period_days": 30,
        "annual_escalation_pct": 5.0,
        "kyc_id_type": "Aadhaar",
        "kyc_id_masked": "XXXX-XXXX-9104",
    },
]


def generate_sample_lease_pdfs(output_dir: Path) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    for meta in SAMPLE_LEASES_METADATA:
        pdf_path = output_dir / meta["filename"]
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4

        # Page 1: Core Lease Terms & Financial Covenants
        y = height - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, f"RESIDENTIAL & COMMERCIAL LEASE AGREEMENT - {meta['lease_id']}")
        y -= 25
        c.setFont("Helvetica", 10)
        lines_p1 = [
            f"Lease ID: {meta['lease_id']}",
            f"Property Name: {meta['property_name']}",
            f"Unit Number: {meta['unit_number']}",
            f"Landlord Entity: {meta['landlord_name']}",
            f"Tenant Name: {meta['tenant_name']}",
            f"KYC Verification Type: {meta['kyc_id_type']}",
            f"KYC Masked Reference: {meta['kyc_id_masked']}",
            f"Lease Start Date: {meta['lease_start_date']}",
            f"Lease End Date: {meta['lease_end_date']}",
            "",
            "CLAUSE 1: RENT AND PAYMENT SCHEDULE",
            f"1.1 The Tenant agrees to pay a Monthly Rent of INR {meta['monthly_rent_inr']:.2f} per month.",
            f"1.2 Rent Due Day: Rent is payable in advance on or before day {meta['due_day_of_month']} of each calendar month.",
            f"1.3 Grace Period: A grace period of {meta['grace_period_days']} calendar days is permitted beyond the due date.",
            f"1.4 Late Fee Penalty: Any rent unpaid after the grace period shall attract a Late Fee of INR {meta['late_fee_inr_per_day']:.2f} per day.",
            "",
            "CLAUSE 2: SECURITY DEPOSIT AND ESCALATION",
            f"2.1 The Tenant has deposited an interest-free refundable Security Deposit of INR {meta['security_deposit_inr']:.2f}.",
            f"2.2 Upon renewal after 12 months, the monthly rent shall increase by an Annual Escalation of {meta['annual_escalation_pct']}%.",
            "",
            "CLAUSE 3: LOCK-IN AND TERMINATION NOTICE",
            f"3.1 Lock-In Period: This agreement enforces a mandatory Lock-In Period of {meta['lock_in_months']} months from commencement.",
            f"3.2 Notice Period: Either party may terminate the lease after the lock-in period by serving {meta['notice_period_days']} days written notice.",
            "3.3 Early vacating during the lock-in period results in forfeiture of the security deposit equivalent to the remaining lock-in rent.",
        ]
        for line in lines_p1:
            if line.startswith("CLAUSE"):
                c.setFont("Helvetica-Bold", 10)
            else:
                c.setFont("Helvetica", 10)
            c.drawString(50, y, line)
            y -= 16

        c.showPage()

        # Page 2: Maintenance SLAs, Liability Allocation & Dispute Resolution
        y = height - 50
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"ANNEXURE A: MAINTENANCE SLAs & LIABILITY MATRIX ({meta['lease_id']})")
        y -= 25
        lines_p2 = [
            "CLAUSE 4: MAINTENANCE SLA TIERS AND RESPONSE WINDOWS",
            "4.1 P0_EMERGENCY (Active water pipe burst, electrical short-circuit, gas leak, structural hazard):",
            "    Mandatory vendor dispatch SLA within 4 hours. Requires immediate human property manager notification.",
            "4.2 P1_HIGH (HVAC/AC breakdown in peak summer, main door lock failure, water heater failure):",
            "    Mandatory resolution SLA within 24 hours.",
            "4.3 P2_MEDIUM (Appliance malfunction, minor seepage, intercom repair):",
            "    Mandatory resolution SLA within 48 hours.",
            "4.4 P3_LOW (Cosmetic paint touch-up, cabinet hinge adjustment, routine pest control):",
            "    Scheduled resolution SLA within 72 hours.",
            "",
            "CLAUSE 5: REPAIR COST LIABILITY ALLOCATION",
            "5.1 Structural, concealed plumbing, main electrical wiring, and external seepage repairs are 100% LANDLORD liability.",
            "5.2 Consumables (light bulbs, fuses, tap washers) and minor wear-and-tear under INR 1,500 are TENANT liability.",
            "5.3 Any single maintenance work order exceeding INR 5,000 requires prior Human-in-the-Loop (HITL) approval from the Property Manager.",
            "",
            "CLAUSE 6: SUBLETTING AND COMPLIANCE",
            "6.1 Subletting or commercial reallocation of residential units without written Landlord consent is strictly prohibited.",
            "6.2 Digital KYC verification and police verification logs must be archived in the TenantEase profile prior to key handover.",
        ]
        for line in lines_p2:
            if line.startswith("CLAUSE"):
                c.setFont("Helvetica-Bold", 10)
            else:
                c.setFont("Helvetica", 10)
            c.drawString(50, y, line)
            y -= 16

        c.save()
        generated_files.append(pdf_path)

    return generated_files
