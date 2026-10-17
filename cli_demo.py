"""
End-to-End CLI Runner & Benchmark Generator for TenantEase AI Engine.
Generates sample lease PDFs, runs all 4 AI/ML pipelines, and writes verified benchmark metrics
to outputs/poc_benchmark_results.json.
"""
import json
from pathlib import Path
from tenantease_ai.pdf_generator import generate_sample_lease_pdfs
from tenantease_ai.risk_and_triage import RentDelinquencyScorer
from tenantease_ai.eval_harness import TenantEaseEvalHarness


def main():
    base = Path(__file__).parent
    pdf_dir = base / "data" / "sample_leases"
    out_dir = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = generate_sample_lease_pdfs(pdf_dir)
    print(f"[1/3] Generated {len(pdfs)} multi-page Lease Agreement PDFs in {pdf_dir}")

    scorer = RentDelinquencyScorer(random_seed=42)
    df, portfolio_metrics = scorer.generate_synthetic_portfolio(n_units=200)
    df.to_csv(out_dir / "tenantease_200_units_risk_scored.csv", index=False)
    print(f"[2/3] Scored 200 TenantEase units -> Accuracy: {portfolio_metrics['accuracy']}, F1: {portfolio_metrics['f1_score']}")

    harness = TenantEaseEvalHarness(pdf_dir)
    eval_report = harness.run_full_suite()
    print(f"[3/3] Executed Evaluation & Red-Teaming Suite -> RAG Pass Rate: {eval_report['rag_pass_rate']*100:.0f}%, Red-Team Block Rate: {eval_report['red_team_block_rate']*100:.0f}%")

    combined = {
        "portfolio_risk_metrics": portfolio_metrics,
        "evaluation_harness_report": eval_report,
    }
    report_path = out_dir / "poc_benchmark_results.json"
    report_path.write_text(json.dumps(combined, indent=2))
    print(f"Saved complete benchmark output to {report_path}")


if __name__ == "__main__":
    main()
