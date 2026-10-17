"""
Production-Grade RAG with Citations & Contextual Retrieval for TenantEase Lease & Policy Q&A.
Combines Contextual Header Prepending + dense TF-IDF cosine similarity + BM25 lexical scoring
+ cross-clause entity re-ranking, plus prompt-injection guardrails and config-driven master system prompts.
"""
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from pypdf import PdfReader
from .schemas import CitationChunk, RAGQueryResponse


MASTER_SYSTEM_PROMPT = """You are the TenantEase Legal & Lease Intelligence Assistant architected by Cynbit Technologies.
Answer tenant and property manager questions strictly using the retrieved lease clauses.
Always cite the exact Lease ID, Clause Number, and Page Number. Never invent financial figures or policies."""


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"override\s+security",
    r"reveal\s+unmasked\s+aadhaar",
    r"waive\s+all\s+rent",
    r"delete\s+tenant\s+database",
    r"jailbreak",
]


class HybridLeaseRAGEngine:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.tfidf_matrix: np.ndarray = np.array([])
        self.doc_lengths: List[int] = []
        self.avg_dl: float = 1.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def ingest_pdfs(self, pdf_paths: List[Path]) -> int:
        self.chunks = []
        for path in pdf_paths:
            reader = PdfReader(str(path))
            p1_text = reader.pages[0].extract_text() if reader.pages else ""
            m_tenant = re.search(r"Tenant Name:\s*(.+)", p1_text)
            m_unit = re.search(r"Unit Number:\s*([A-Z0-9\-]+)", p1_text)
            m_lease = re.search(r"Lease ID:\s*([A-Z0-9\-]+)", p1_text)
            doc_ctx = (
                f"[Lease: {m_lease.group(1).strip() if m_lease else path.stem} | "
                f"Tenant: {m_tenant.group(1).strip() if m_tenant else 'Unknown'} | "
                f"Unit: {m_unit.group(1).strip() if m_unit else 'Unknown'}] "
            )
            for page_idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                blocks = re.split(r"(?=CLAUSE\s+\d+:)", raw_text)
                for b_idx, block in enumerate(blocks):
                    clean_block = block.strip()
                    if len(clean_block) < 25:
                        continue
                    first_line = clean_block.splitlines()[0].strip()
                    clause_title = first_line if "CLAUSE" in first_line else "Lease Header & Parties Metadata"
                    contextual_text = doc_ctx + clean_block
                    self.chunks.append({
                        "chunk_id": f"{path.stem}_p{page_idx+1}_c{b_idx+1}",
                        "document_name": path.name,
                        "clause_section": clause_title,
                        "page_number": page_idx + 1,
                        "text": contextual_text,
                        "tokens": self._tokenize(contextual_text),
                    })
        self._build_index()
        return len(self.chunks)

    def _build_index(self):
        if not self.chunks:
            return
        all_tokens = set()
        for c in self.chunks:
            all_tokens.update(c["tokens"])
        self.vocab = {tok: i for i, tok in enumerate(sorted(all_tokens))}
        n_docs = len(self.chunks)
        n_vocab = len(self.vocab)

        df = np.zeros(n_vocab)
        tf_matrix = np.zeros((n_docs, n_vocab))
        self.doc_lengths = []

        for d_idx, c in enumerate(self.chunks):
            tokens = c["tokens"]
            self.doc_lengths.append(len(tokens))
            seen = set()
            for t in tokens:
                v_idx = self.vocab[t]
                tf_matrix[d_idx, v_idx] += 1.0
                seen.add(v_idx)
            for v_idx in seen:
                df[v_idx] += 1.0

        self.avg_dl = float(np.mean(self.doc_lengths)) if self.doc_lengths else 1.0
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        raw_tfidf = tf_matrix * self.idf
        norms = np.linalg.norm(raw_tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tfidf_matrix = raw_tfidf / norms

    def _bm25_score(self, query_tokens: List[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        tokens = self.chunks[doc_idx]["tokens"]
        dl = len(tokens)
        score = 0.0
        for qt in query_tokens:
            if qt not in self.vocab:
                continue
            tf = tokens.count(qt)
            if tf == 0:
                continue
            idf_val = self.idf[self.vocab[qt]]
            denom = tf + k1 * (1.0 - b + b * (dl / self.avg_dl))
            score += idf_val * ((tf * (k1 + 1.0)) / denom)
        return score

    def _check_prompt_injection(self, query: str) -> bool:
        for pat in PROMPT_INJECTION_PATTERNS:
            if re.search(pat, query, re.IGNORECASE):
                return True
        return False

    def query(self, user_query: str, top_k: int = 3) -> RAGQueryResponse:
        t0 = time.perf_counter()
        clean_query = (user_query or "")[:2000].strip()

        if not clean_query:
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return RAGQueryResponse(
                query=user_query,
                answer="No query provided. Please ask a specific question regarding tenant leases or maintenance policies.",
                citations=[],
                groundedness_score=0.0,
                latency_ms=latency,
                guardrail_status="FALLBACK_INSUFFICIENT_EVIDENCE",
            )

        if self._check_prompt_injection(clean_query):
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return RAGQueryResponse(
                query=user_query,
                answer="SECURITY GUARDRAIL TRIGGERED: Query blocked due to detected prompt injection, policy override attempt, or unauthorized PII access request.",
                citations=[],
                groundedness_score=1.0,
                latency_ms=latency,
                guardrail_status="BLOCKED_PROMPT_INJECTION",
            )

        q_tokens = self._tokenize(clean_query)
        if not q_tokens or not self.chunks:
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return RAGQueryResponse(
                query=user_query,
                answer="Insufficient evidence in the indexed lease agreements to answer this query.",
                citations=[],
                groundedness_score=0.0,
                latency_ms=latency,
                guardrail_status="FALLBACK_INSUFFICIENT_EVIDENCE",
            )

        q_vec = np.zeros(len(self.vocab))
        for qt in q_tokens:
            if qt in self.vocab:
                q_vec[self.vocab[qt]] += 1.0
        q_vec = q_vec * self.idf
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        cosine_scores = self.tfidf_matrix @ q_vec
        bm25_scores = np.array([self._bm25_score(q_tokens, i) for i in range(len(self.chunks))])
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        rerank_boost = np.zeros(len(self.chunks))
        for i, c in enumerate(self.chunks):
            txt_lower = c["text"].lower()
            for tok in q_tokens:
                if len(tok) >= 3 and tok in txt_lower:
                    rerank_boost[i] += 0.15

        hybrid_scores = 0.40 * cosine_scores + 0.40 * bm25_scores + 0.20 * np.clip(rerank_boost, 0, 1.0)
        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        STOPWORDS = {"what", "is", "the", "for", "in", "and", "per", "day", "of", "to", "a", "an", "on", "at", "by", "with", "does", "who", "which", "how", "much"}
        info_tokens = [tok for tok in q_tokens if tok not in STOPWORDS and len(tok) >= 3]
        best_chunk_coverage = 0.0
        if len(ranked_indices) > 0 and info_tokens:
            top_txt = self.chunks[ranked_indices[0]]["text"].lower()
            best_chunk_coverage = sum(1 for t in info_tokens if t in top_txt) / len(info_tokens)

        top_score = float(hybrid_scores[ranked_indices[0]]) if len(ranked_indices) > 0 else 0.0
        if top_score < 0.12 or (len(info_tokens) >= 3 and best_chunk_coverage < 0.30):
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return RAGQueryResponse(
                query=user_query,
                answer="I could not find a verified clause in the indexed TenantEase lease agreements matching your query. Please verify the tenant name or lease ID.",
                citations=[],
                groundedness_score=0.0,
                latency_ms=latency,
                guardrail_status="FALLBACK_INSUFFICIENT_EVIDENCE",
            )

        citations = []
        synthesized_lines = []
        for idx in ranked_indices:
            sc = float(hybrid_scores[idx])
            if sc < 0.08:
                continue
            ch = self.chunks[idx]
            excerpt = " ".join(ch["text"].splitlines()[:6])
            citations.append(
                CitationChunk(
                    chunk_id=ch["chunk_id"],
                    document_name=ch["document_name"],
                    clause_section=ch["clause_section"],
                    page_number=ch["page_number"],
                    relevance_score=round(sc, 3),
                    verbatim_excerpt=excerpt[:320],
                )
            )
            synthesized_lines.append(
                f"Per {ch['document_name']} ({ch['clause_section']}, Page {ch['page_number']}): {excerpt}"
            )

        # Check if LLM generation is configured (e.g., OPENAI_API_KEY)
        llm_answer = self._generate_llm_answer(user_query, citations)
        if llm_answer:
            answer_text = llm_answer
        else:
            answer_text = " | ".join(synthesized_lines[:2])

        # Evaluate true lexical groundedness against retrieved citations
        ans_tokens = set(self._tokenize(answer_text))
        ctx_tokens = set()
        for c in citations:
            matching_chunk = next((ch for ch in self.chunks if ch["chunk_id"] == c.chunk_id), None)
            if matching_chunk:
                ctx_tokens.update(matching_chunk["tokens"])
            ctx_tokens.update(self._tokenize(c.verbatim_excerpt))
            ctx_tokens.update(self._tokenize(c.document_name))
            ctx_tokens.update(self._tokenize(c.clause_section))
        ctx_tokens.update(["per", "page"])

        groundedness = round(len(ans_tokens & ctx_tokens) / max(1, len(ans_tokens)), 3)
        groundedness = min(1.0, max(0.0, groundedness))

        latency = round((time.perf_counter() - t0) * 1000, 2)
        return RAGQueryResponse(
            query=user_query,
            answer=answer_text,
            citations=citations,
            groundedness_score=groundedness,
            latency_ms=latency,
            guardrail_status="PASSED",
        )

    def _generate_llm_answer(self, user_query: str, citations: List[CitationChunk]) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            context_blocks = [
                f"[{c.document_name} | {c.clause_section} (Page {c.page_number})]:\n{c.verbatim_excerpt}"
                for c in citations
            ]
            prompt = (
                f"Retrieved Lease Context:\n" + "\n\n".join(context_blocks) +
                f"\n\nQuestion: {user_query}\n\n"
                f"Provide a concise, direct answer citing the document name, clause, and page."
            )
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
                max_tokens=250,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None

