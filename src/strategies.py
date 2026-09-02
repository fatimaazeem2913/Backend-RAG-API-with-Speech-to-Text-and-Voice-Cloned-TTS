import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class RetrievalStrategyManager:
    """Unified manager implementing Dense, BM25, Hybrid (RRF), and Hierarchical Compression retrieval."""

    def __init__(
        self,
        persist_dir: str = "./outputs/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_docs: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
        self._initialize_from_corpus()

    def get_indexed_sources(self) -> List[Dict[str, Any]]:
        """Returns a list of unique indexed documents and their chunk counts directly from ChromaDB."""
        if not self.vectorstore or self.vectorstore._collection.count() == 0:
            return []
        
        try:
            # Retrieve all metadata from the ChromaDB collection
            results = self.vectorstore._collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            # Aggregate chunk counts by distinct filename/source
            source_counts = {}
            for meta in metadatas:
                if not meta:
                    continue
                source = meta.get("source", "unknown_document")
                source_counts[source] = source_counts.get(source, 0) + 1
                
            return [{"filename": src, "total_chunks": count} for src, count in source_counts.items()]
            
        except Exception as e:
            print(f"[Strategy Manager] Warning - Error extracting source list: {e}")
            return [{"filename": "Indexed Corpus", "total_chunks": self.vectorstore._collection.count()}]

    def _initialize_from_corpus(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="day21_enterprise_rag"
            )
            if self.vectorstore._collection.count() == 0:
                self._load_fallback_data()
            else:
                self._rebuild_bm25_from_vectorstore()
        except Exception:
            self._load_fallback_data()

    def _load_fallback_data(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(current_dir)
        root_dir = os.path.dirname(backend_dir)

        candidate_paths = [
            os.path.join(backend_dir, "data/sample_corpus.json"),
            os.path.join(root_dir, "day-20-conversational-rag/data/sample_corpus.json"),
            os.path.join(root_dir, "day-19-hybrid-search-advanced-retrieval/data/sample_corpus.json"),
            os.path.join(root_dir, "day-16-chunking-strategies/outputs/chunks_hierarchical.json")
        ]
        
        chunks = []
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            chunks = data
                            print(f"[Corpus Loader] Loaded {len(chunks)} chunks from {p}")
                            break
                except Exception:
                    continue

        docs = []
        for c in chunks:
            # Handle heterogeneous chunk key schemas
            text = c.get("text") or c.get("content") or c.get("page_content") or c.get("chunk_text") or ""
            if not text.strip():
                continue

            meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}
            
            # Extract root level overrides
            source = c.get("source") or meta.get("source") or meta.get("file_name") or "SupportcoursesM-DLearning.pdf"
            page = c.get("page") or meta.get("page") or meta.get("page_number") or 1
            chunk_id = c.get("chunk_id") or c.get("id") or meta.get("chunk_id") or f"{source}_p{page}"

            clean_meta = {
                "source": str(source),
                "page": int(page) if str(page).isdigit() else 1,
                "chunk_id": str(chunk_id)
            }
            docs.append(Document(page_content=text, metadata=clean_meta))

        if docs:
            self.index_documents(docs)

    def _rebuild_bm25_from_vectorstore(self):
        try:
            raw_data = self.vectorstore.get()
            docs = []
            for text, meta in zip(raw_data.get("documents", []), raw_data.get("metadatas", [])):
                docs.append(Document(page_content=text, metadata=meta or {}))
            self.corpus_docs = docs
            self.tokenized_corpus = [doc.page_content.lower().split() for doc in self.corpus_docs]
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
        except Exception:
            pass

    def index_documents(self, docs: List[Document]):
        if not docs:
            return
        self.corpus_docs = list(docs)
        self.tokenized_corpus = [doc.page_content.lower().split() for doc in self.corpus_docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

        self.vectorstore = Chroma.from_documents(
            documents=self.corpus_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="day21_enterprise_rag"
        )

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.vectorstore or self.vectorstore._collection.count() == 0:
            self._load_fallback_data()
        if not self.vectorstore or self.vectorstore._collection.count() == 0:
            return []
        return self.vectorstore.similarity_search(query, k=top_k)

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.bm25 or not self.corpus_docs:
            self._load_fallback_data()
        if not self.bm25 or not self.corpus_docs:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.corpus_docs[i] for i in top_indices if scores[i] > 0.0]

    def retrieve_hybrid_rrf(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Document]:
        dense_docs = self.retrieve_dense(query, top_k=top_k * 2)
        sparse_docs = self.retrieve_bm25(query, top_k=top_k * 2)

        if not dense_docs and not sparse_docs:
            return []

        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            doc_id = str(doc.metadata.get("chunk_id", hash(doc.page_content)))
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, doc in enumerate(sparse_docs):
            doc_id = str(doc.metadata.get("chunk_id", hash(doc.page_content)))
            doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)[:top_k]
        return [doc_map[did] for did in sorted_ids]

    def retrieve_hierarchical(self, query: str, top_k: int = 5) -> List[Document]:
        candidates = self.retrieve_hybrid_rrf(query, top_k=top_k)
        compressed = []
        query_terms = {t.lower() for t in query.split() if len(t) > 2}

        for doc in candidates:
            sentences = [s.strip() for s in doc.page_content.split(".") if s.strip()]
            selected = [s for s in sentences if any(t in s.lower() for t in query_terms)]
            # Preserve minimum context fidelity
            content = ". ".join(selected) + "." if len(selected) >= 2 else doc.page_content
            compressed.append(Document(page_content=content, metadata=doc.metadata))
        return compressed