"""
J.A.R.V.I.S — Intelligent Documentation RAG

A single-file, source-grounded RAG engine for technical documentation.
Designed to live independently in the repository without modifying existing code.

Features
--------
- PDF, DOCX, Markdown, TXT, HTML and JSON ingestion
- Recursive semantic-aware chunking with overlap
- Rich metadata: software, version, module, document, section, page, path
- Persistent SQLite vector store (embeddings stored as JSON; no external DB required)
- Sentence-Transformers semantic embeddings when installed
- Pure-Python BM25 lexical retrieval
- Hybrid semantic + lexical retrieval with reciprocal-rank fusion
- Optional CrossEncoder reranking
- Version-aware retrieval and conflict avoidance
- Exact-source citations and page/section tracking
- Evidence-first / anti-hallucination answer prompt builder
- FastAPI endpoint when FastAPI is installed
- CLI for ingest/search

Recommended optional dependencies
---------------------------------
pip install pypdf python-docx beautifulsoup4 numpy sentence-transformers fastapi uvicorn

The core lexical search and persistence work with the Python standard library.
Semantic search requires sentence-transformers + numpy. PDF/DOCX ingestion requires
pypdf/python-docx respectively.

Important
---------
This file does NOT call an LLM by itself. It retrieves and verifies evidence and can
build a grounded prompt for an LLM or agent. This makes it suitable as a tool behind
ChatGPT/J.A.R.V.I.S rather than pretending that retrieval itself is generation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sqlite3
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("JARVIS_RAG_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_DB = os.getenv("JARVIS_RAG_DB", ".jarvis_rag.sqlite3")
DEFAULT_CHUNK_SIZE = int(os.getenv("JARVIS_RAG_CHUNK_SIZE", "1200"))
DEFAULT_OVERLAP = int(os.getenv("JARVIS_RAG_OVERLAP", "180"))

SOFTWARE_ALIASES = {
    "schrodinger": "Schrödinger",
    "schrodinger suite": "Schrödinger",
    "maestro": "Schrödinger",
    "jaguar": "Schrödinger",
    "desmond": "Schrödinger",
    "materials science": "Schrödinger",
    "orca": "ORCA",
    "material studio": "Materials Studio",
    "materials studio": "Materials Studio",
    "hysys": "Aspen HYSYS",
    "aspen hysys": "Aspen HYSYS",
}

VERSION_PATTERNS = [
    re.compile(r"\b(?:version|ver\.?|release|v)\s*([0-9]+(?:\.[0-9]+){1,3})\b", re.I),
    re.compile(r"\b(20[0-9]{2}\.[0-9]+)\b"),
    re.compile(r"\b(20[0-9]{2})\b"),
]

TOKEN_RE = re.compile(r"[\w./:+#-]+", re.UNICODE)
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_path: str
    document_title: str
    software: Optional[str] = None
    version: Optional[str] = None
    module: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        bits = [self.document_title]
        if self.section:
            bits.append(f"§ {self.section}")
        if self.page is not None:
            bits.append(f"p. {self.page}")
        if self.version:
            bits.append(f"v{self.version}")
        return " — ".join(bits)

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    chunk: DocumentChunk
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: Optional[float] = None
    evidence_score: float = 0.0

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.evidence_score


# ---------------------------------------------------------------------------
# Text and metadata utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_software(text: str, path: str = "") -> Optional[str]:
    haystack = f"{path}\n{text[:12000]}".lower()
    # Longest aliases first to avoid matching a short alias prematurely.
    for alias, canonical in sorted(SOFTWARE_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in haystack:
            return canonical
    return None


def detect_version(text: str, path: str = "") -> Optional[str]:
    haystack = f"{path}\n{text[:16000]}"
    for pattern in VERSION_PATTERNS:
        match = pattern.search(haystack)
        if match:
            return match.group(1)
    return None


def detect_module(text: str) -> Optional[str]:
    known = [
        "Maestro", "Jaguar", "Desmond", "Materials Science", "QSite",
        "Glide", "Prime", "LigPrep", "MacroModel", "ORCA", "HYSYS",
        "Forcite", "CASTEP", "DMol3", "Amorphous Cell",
    ]
    haystack = text[:10000].lower()
    for item in known:
        if item.lower() in haystack:
            return item
    return None


def document_title(path: str, text: str) -> str:
    for line in text.splitlines()[:40]:
        m = HEADING_RE.match(line)
        if m:
            return m.group(2).strip()
    return Path(path).stem.replace("_", " ").replace("-", " ").strip() or Path(path).name


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(unicodedata.normalize("NFKC", text))]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_file(path: Path) -> List[Tuple[str, Optional[int]]]:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".markdown", ".rst", ".csv", ".json"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]

    if suffix in {".html", ".htm"}:
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return [(soup.get_text("\n"), None)]
        except ImportError:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            raw = re.sub(r"<[^>]+>", " ", raw)
            return [(html.unescape(raw), None)]

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires: pip install pypdf") from exc
        reader = PdfReader(str(path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            pages.append((page.extract_text() or "", number))
        return pages

    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore
        except ImportError as exc:
            raise RuntimeError("DOCX ingestion requires: pip install python-docx") from exc
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return [(text, None)]

    raise ValueError(f"Unsupported document type: {suffix} ({path})")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class SmartChunker:
    """Heading-aware chunker that tries to keep procedures and parameter blocks intact."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[Tuple[str, Optional[str]]]:
        text = normalize_text(text)
        if not text:
            return []

        sections: List[Tuple[Optional[str], str]] = []
        current_heading: Optional[str] = None
        current: List[str] = []

        for line in text.splitlines():
            heading = HEADING_RE.match(line)
            if heading:
                if current:
                    sections.append((current_heading, "\n".join(current).strip()))
                    current = []
                current_heading = heading.group(2).strip()
                current.append(line)
            else:
                current.append(line)
        if current:
            sections.append((current_heading, "\n".join(current).strip()))

        chunks: List[Tuple[str, Optional[str]]] = []
        for heading, section_text in sections:
            if len(section_text) <= self.chunk_size:
                if section_text:
                    chunks.append((section_text, heading))
                continue

            paragraphs = re.split(r"\n\s*\n", section_text)
            buf = ""
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                candidate = f"{buf}\n\n{paragraph}".strip() if buf else paragraph
                if len(candidate) <= self.chunk_size:
                    buf = candidate
                    continue
                if buf:
                    chunks.append((buf, heading))
                # Long single paragraphs are split on sentence-ish boundaries.
                while len(paragraph) > self.chunk_size:
                    cut = self._best_cut(paragraph[: self.chunk_size])
                    piece = paragraph[:cut].strip()
                    if piece:
                        chunks.append((piece, heading))
                    paragraph = paragraph[max(0, cut - self.overlap):].strip()
                buf = paragraph
            if buf:
                chunks.append((buf, heading))

        return chunks

    @staticmethod
    def _best_cut(text: str) -> int:
        for marker in ["\n", ". ", "; ", ", ", " "]:
            pos = text.rfind(marker)
            if pos >= int(len(text) * 0.55):
                return max(1, pos + (1 if marker != "\n" else 0))
        return len(text)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def available(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic retrieval requires sentence-transformers and numpy. "
                    "Install with: pip install sentence-transformers numpy"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._load()
        vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# SQLite vector + document store
# ---------------------------------------------------------------------------

class SQLiteStore:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source_path TEXT NOT NULL,
                document_title TEXT NOT NULL,
                software TEXT,
                version TEXT,
                module TEXT,
                section TEXT,
                page INTEGER,
                chunk_index INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT,
                content_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_software ON chunks(software);
            CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(version);
            CREATE INDEX IF NOT EXISTS idx_chunks_module ON chunks(module);
            CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
            """
        )
        self.conn.commit()

    def upsert(self, chunks: Sequence[DocumentChunk], embeddings: Optional[Sequence[Sequence[float]]] = None):
        rows = []
        for i, chunk in enumerate(chunks):
            vector = list(embeddings[i]) if embeddings is not None else None
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.text,
                    chunk.source_path,
                    chunk.document_title,
                    chunk.software,
                    chunk.version,
                    chunk.module,
                    chunk.section,
                    chunk.page,
                    chunk.chunk_index,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    json.dumps(vector) if vector is not None else None,
                    sha256_text(chunk.text),
                )
            )
        self.conn.executemany(
            """
            INSERT INTO chunks
            (chunk_id,text,source_path,document_title,software,version,module,section,page,
             chunk_index,metadata_json,embedding_json,content_hash)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                text=excluded.text, source_path=excluded.source_path,
                document_title=excluded.document_title, software=excluded.software,
                version=excluded.version, module=excluded.module, section=excluded.section,
                page=excluded.page, chunk_index=excluded.chunk_index,
                metadata_json=excluded.metadata_json, embedding_json=excluded.embedding_json,
                content_hash=excluded.content_hash
            """,
            rows,
        )
        self.conn.commit()

    def all_rows(self) -> List[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM chunks"))

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def close(self):
        self.conn.close()


def row_to_chunk(row: sqlite3.Row) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=row["chunk_id"],
        text=row["text"],
        source_path=row["source_path"],
        document_title=row["document_title"],
        software=row["software"],
        version=row["version"],
        module=row["module"],
        section=row["section"],
        page=row["page"],
        chunk_index=row["chunk_index"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


# ---------------------------------------------------------------------------
# BM25 lexical retrieval
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, rows: Sequence[sqlite3.Row]):
        self.rows = list(rows)
        self.docs = [tokenize(r["text"]) for r in self.rows]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.df: Dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.k1 = 1.5
        self.b = 0.75

    def score(self, query: str) -> List[float]:
        qterms = tokenize(query)
        n = len(self.docs)
        scores = [0.0] * n
        for i, doc in enumerate(self.docs):
            if not doc:
                continue
            tf: Dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            for term in qterms:
                if term not in tf:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * len(doc) / max(1, self.avgdl))
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores


# ---------------------------------------------------------------------------
# Main RAG engine
# ---------------------------------------------------------------------------

class IntelligentRAG:
    """
    Evidence-first technical-documentation RAG.

    The engine is deliberately model-agnostic: retrieval is deterministic and
    the final LLM call can be made by the host agent.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        embedding_model: str = DEFAULT_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        self.store = SQLiteStore(db_path)
        self.embedder = EmbeddingModel(embedding_model)
        self.chunker = SmartChunker(chunk_size, overlap)

    # --------------------------- ingestion ---------------------------

    def ingest(self, path: str, recursive: bool = True, force: bool = False) -> Dict[str, Any]:
        root = Path(path)
        files = [root] if root.is_file() else list(root.rglob("*")) if recursive else list(root.glob("*"))
        files = [p for p in files if p.is_file() and p.suffix.lower() in {
            ".pdf", ".docx", ".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".json", ".csv"
        }]

        total_chunks = 0
        processed = 0
        skipped = 0
        errors: List[Dict[str, str]] = []

        for file_path in files:
            try:
                pages = parse_file(file_path)
                full_text = normalize_text("\n\n".join(text for text, _ in pages))
                if not full_text:
                    skipped += 1
                    continue

                title = document_title(str(file_path), full_text)
                software = detect_software(full_text, str(file_path))
                version = detect_version(full_text, str(file_path))
                module = detect_module(full_text)
                chunks: List[DocumentChunk] = []

                # Keep PDF page boundaries so citations remain useful.
                for page_text, page in pages:
                    for local_index, (text, section) in enumerate(self.chunker.split(page_text)):
                        seed = f"{file_path}|{page}|{local_index}|{sha256_text(text)}"
                        chunk_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                text=text,
                                source_path=str(file_path),
                                document_title=title,
                                software=software,
                                version=version,
                                module=module,
                                section=section,
                                page=page,
                                chunk_index=local_index,
                                metadata={
                                    "extension": file_path.suffix.lower(),
                                    "absolute_path": str(file_path.resolve()),
                                },
                            )
                        )

                embeddings = self.embedder.encode([c.text for c in chunks]) if self.embedder.available() else None
                self.store.upsert(chunks, embeddings)
                total_chunks += len(chunks)
                processed += 1
            except Exception as exc:  # keep batch ingestion alive
                errors.append({"file": str(file_path), "error": str(exc)})

        return {
            "processed_files": processed,
            "skipped_files": skipped,
            "chunks_indexed": total_chunks,
            "errors": errors,
            "semantic_embeddings": self.embedder.available(),
            "database": self.store.db_path,
        }

    # --------------------------- filtering ---------------------------

    @staticmethod
    def _matches_filters(chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        for key in ("software", "version", "module", "source_path"):
            expected = filters.get(key)
            if expected is None:
                continue
            actual = getattr(chunk, key, None)
            if actual is None:
                return False
            if str(actual).lower() != str(expected).lower():
                return False
        return True

    @staticmethod
    def infer_filters(query: str) -> Dict[str, Any]:
        software = detect_software(query)
        version = detect_version(query)
        module = detect_module(query)
        filters: Dict[str, Any] = {}
        if software:
            filters["software"] = software
        if version:
            filters["version"] = version
        if module:
            filters["module"] = module
        return filters

    # --------------------------- retrieval ---------------------------

    def search(
        self,
        query: str,
        top_k: int = 8,
        candidate_k: int = 40,
        filters: Optional[Dict[str, Any]] = None,
        require_version: bool = False,
        rerank: bool = True,
    ) -> List[SearchResult]:
        rows = self.store.all_rows()
        if not rows:
            return []

        inferred = self.infer_filters(query)
        filters = {**inferred, **(filters or {})}
        query_version = filters.get("version")

        candidates: List[sqlite3.Row] = []
        for row in rows:
            chunk = row_to_chunk(row)
            if not self._matches_filters(chunk, filters):
                continue
            if require_version and query_version and chunk.version != query_version:
                continue
            candidates.append(row)

        # If strict filters eliminated everything, fail safely rather than
        # silently returning a different software/version.
        if not candidates:
            return []

        bm25 = BM25(candidates)
        lexical = bm25.score(query)

        semantic_scores = [0.0] * len(candidates)
        query_vector: Optional[List[float]] = None
        if self.embedder.available():
            try:
                query_vector = self.embedder.encode_one(query)
            except Exception:
                query_vector = None
        if query_vector is not None:
            for i, row in enumerate(candidates):
                stored = json.loads(row["embedding_json"] or "null")
                if stored:
                    semantic_scores[i] = max(0.0, cosine(query_vector, stored))

        lex_order = sorted(range(len(candidates)), key=lambda i: lexical[i], reverse=True)
        sem_order = sorted(range(len(candidates)), key=lambda i: semantic_scores[i], reverse=True)
        lex_rank = {idx: rank + 1 for rank, idx in enumerate(lex_order)}
        sem_rank = {idx: rank + 1 for rank, idx in enumerate(sem_order)}

        results: List[SearchResult] = []
        for i, row in enumerate(candidates):
            # Reciprocal Rank Fusion is robust when lexical and semantic scores
            # live on different scales.
            fusion = 0.55 / (60 + lex_rank[i]) + 0.45 / (60 + sem_rank[i])
            chunk = row_to_chunk(row)
            results.append(
                SearchResult(
                    chunk=chunk,
                    semantic_score=semantic_scores[i],
                    lexical_score=lexical[i],
                    fusion_score=fusion,
                    evidence_score=fusion,
                )
            )

        results.sort(key=lambda r: r.fusion_score, reverse=True)
        results = results[: max(top_k, candidate_k)]

        if rerank:
            results = self._cross_encoder_rerank(query, results, top_k)
        else:
            results = results[:top_k]

        # Evidence quality boost: exact version + software matches and source
        # specificity should outrank generic matches when scores are close.
        for result in results:
            result.evidence_score = result.fusion_score
            if query_version and result.chunk.version == query_version:
                result.evidence_score += 0.03
            if filters.get("software") and result.chunk.software == filters["software"]:
                result.evidence_score += 0.02
            if result.chunk.section:
                result.evidence_score += 0.005
        results.sort(key=lambda r: r.evidence_score, reverse=True)
        return results[:top_k]

    def _cross_encoder_rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            model_name = os.getenv(
                "JARVIS_RERANK_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
            model = CrossEncoder(model_name)
            pairs = [(query, r.chunk.text) for r in results]
            scores = model.predict(pairs)
            for result, score in zip(results, scores):
                result.rerank_score = float(score)
            results.sort(key=lambda r: r.rerank_score or -1e9, reverse=True)
            return results[:top_k]
        except Exception:
            # Reranking is an enhancement, never a reason for retrieval to fail.
            return results[:top_k]

    # --------------------------- grounded prompt ---------------------------

    def build_grounded_prompt(
        self,
        query: str,
        results: Sequence[SearchResult],
        strict: bool = True,
    ) -> str:
        if not results:
            return (
                "No authoritative documentation evidence was retrieved. "
                "Do not invent parameters, syntax, values, or procedures. "
                "State that the answer was not found in the indexed documentation."
            )

        evidence_blocks = []
        for i, result in enumerate(results, start=1):
            c = result.chunk
            evidence_blocks.append(
                f"[SOURCE {i}]\n"
                f"Citation: {c.citation()}\n"
                f"Path: {c.source_path}\n"
                f"Software: {c.software or 'unknown'}\n"
                f"Version: {c.version or 'unknown'}\n"
                f"Module: {c.module or 'unknown'}\n"
                f"Section: {c.section or 'unknown'}\n"
                f"Page: {c.page if c.page is not None else 'unknown'}\n"
                f"Evidence:\n{c.text}"
            )

        rules = [
            "Answer using the retrieved evidence as the primary authority.",
            "Cite the source number for factual claims.",
            "Preserve exact syntax, parameter names, and values when documented.",
            "Do not silently substitute a different software version.",
            "Clearly label any inference; do not present inference as documentation.",
        ]
        if strict:
            rules += [
                "If the requested fact is not supported by the evidence, say: NOT FOUND IN INDEXED DOCUMENTATION.",
                "Never fabricate configuration parameters, file syntax, paths, or numeric settings.",
            ]

        return (
            "You are an evidence-grounded technical documentation assistant.\n\n"
            "USER QUESTION:\n" + query + "\n\n"
            "RULES:\n- " + "\n- ".join(rules) + "\n\n"
            "RETRIEVED EVIDENCE:\n" + "\n\n".join(evidence_blocks)
        )

    def answer_context(self, query: str, top_k: int = 8, **kwargs: Any) -> Dict[str, Any]:
        results = self.search(query, top_k=top_k, **kwargs)
        return {
            "query": query,
            "filters": self.infer_filters(query),
            "found": bool(results),
            "sources": [
                {
                    "rank": i,
                    "citation": r.chunk.citation(),
                    "path": r.chunk.source_path,
                    "software": r.chunk.software,
                    "version": r.chunk.version,
                    "module": r.chunk.module,
                    "section": r.chunk.section,
                    "page": r.chunk.page,
                    "semantic_score": round(r.semantic_score, 6),
                    "lexical_score": round(r.lexical_score, 6),
                    "fusion_score": round(r.fusion_score, 6),
                    "evidence_score": round(r.evidence_score, 6),
                    "text": r.chunk.text,
                }
                for i, r in enumerate(results, start=1)
            ],
            "grounded_prompt": self.build_grounded_prompt(query, results),
        }


# ---------------------------------------------------------------------------
# Optional HTTP API
# ---------------------------------------------------------------------------

def create_app(rag: Optional[IntelligentRAG] = None):
    try:
        from fastapi import FastAPI, HTTPException  # type: ignore
        from pydantic import BaseModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("API requires: pip install fastapi uvicorn") from exc

    engine = rag or IntelligentRAG()
    app = FastAPI(title="J.A.R.V.I.S Intelligent RAG", version="1.0.0")

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 8
        require_version: bool = False
        filters: Optional[Dict[str, Any]] = None

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "chunks": engine.store.count(),
            "semantic_embeddings": engine.embedder.available(),
        }

    @app.post("/search")
    def search_endpoint(request: SearchRequest):
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query cannot be empty")
        return engine.answer_context(
            request.query,
            top_k=max(1, min(request.top_k, 50)),
            require_version=request.require_version,
            filters=request.filters,
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> int:
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S Intelligent Documentation RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Index documents")
    ingest.add_argument("path")
    ingest.add_argument("--db", default=DEFAULT_DB)
    ingest.add_argument("--no-recursive", action="store_true")

    search = sub.add_parser("search", help="Search indexed documentation")
    search.add_argument("query")
    search.add_argument("--db", default=DEFAULT_DB)
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--no-rerank", action="store_true")
    search.add_argument("--strict-version", action="store_true")

    api = sub.add_parser("api", help="Run FastAPI server")
    api.add_argument("--db", default=DEFAULT_DB)
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "ingest":
        rag = IntelligentRAG(db_path=args.db)
        result = rag.ingest(args.path, recursive=not args.no_recursive)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not result["errors"] else 2

    if args.command == "search":
        rag = IntelligentRAG(db_path=args.db)
        result = rag.answer_context(
            args.query,
            top_k=args.top_k,
            rerank=not args.no_rerank,
            require_version=args.strict_version,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "api":
        try:
            import uvicorn  # type: ignore
        except ImportError:
            print("Install API dependencies with: pip install fastapi uvicorn", file=sys.stderr)
            return 2
        app = create_app(IntelligentRAG(db_path=args.db))
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
