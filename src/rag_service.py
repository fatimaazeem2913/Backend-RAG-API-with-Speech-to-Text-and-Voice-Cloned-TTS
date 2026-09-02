import os
import re
import json
from typing import List, Dict, Any, Generator
from google import genai
from dotenv import load_dotenv
from src.strategies import RetrievalStrategyManager

load_dotenv()

class EnterpriseRAGService:
    def __init__(self):
        self.strategy_mgr = RetrievalStrategyManager()
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.models_to_try = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.7-flash"]
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        sid = session_id or "default_session"
        if sid not in self.sessions:
            self.sessions[sid] = []
        return self.sessions[sid]

    def reset_session(self, session_id: str):
        sid = session_id or "default_session"
        if sid in self.sessions:
            self.sessions[sid] = []

    def contextualize(self, session_id: str, query: str) -> str:
        history = self.get_history(session_id)
        pronouns = ["it", "its", "they", "them", "their", "this", "that", "these", "those"]
        has_pronoun = any(re.search(rf"\b{p}\b", query.lower()) for p in pronouns)

        if not history or not has_pronoun or not self.client:
            return query

        valid_turns = [
            t for t in history[-4:]
            if "The provided documentation does not contain information" not in t.get("content", "")
        ]
        if not valid_turns:
            return query

        history_str = "\n".join([f"{t['role'].capitalize()}: {t['content'][:250]}" for t in valid_turns])
        prompt = f"""Given the following conversation history and follow-up question, rewrite the follow-up question into a complete, standalone technical search query. Resolve all ambiguous pronouns ('it', 'its', 'they'). Do NOT answer the question, only output the standalone query.

Conversation History:
{history_str}

Follow-up Question: {query}
Standalone Query:"""

        for m in self.models_to_try:
            try:
                res = self.client.models.generate_content(model=m, contents=prompt)
                if res and res.text:
                    rewritten = res.text.strip().strip("\"'")
                    if rewritten.lower().startswith("standalone query:"):
                        rewritten = rewritten.split(":", 1)[1].strip()
                    return rewritten
            except Exception:
                continue
        return query

    def _retrieve_and_build_prompt(self, sid: str, message: str, strategy: str):
        """Shared by chat() and chat_stream() so retrieval/prompt logic can't drift out of sync."""
        standalone = self.contextualize(sid, message)

        if strategy == "dense":
            docs = self.strategy_mgr.retrieve_dense(standalone, top_k=4)
        elif strategy == "bm25":
            docs = self.strategy_mgr.retrieve_bm25(standalone, top_k=4)
        elif strategy == "hierarchical":
            docs = self.strategy_mgr.retrieve_hierarchical(standalone, top_k=4)
        else:
            docs = self.strategy_mgr.retrieve_hybrid_rrf(standalone, top_k=4)

        ctx_blocks = [
            f"[Source: {d.metadata.get('source', 'doc')}, Page: {d.metadata.get('page', 1)}]\n{d.page_content}"
            for d in docs
        ]
        formatted_ctx = "\n\n---\n\n".join(ctx_blocks) if ctx_blocks else "No relevant context found."
        citations = sorted(list(set([f"[Source: {d.metadata.get('source', 'doc')}, Page: {d.metadata.get('page', 1)}]" for d in docs])))

        qa_prompt = f"""You are an enterprise AI technical assistant.
Answer the user's question clearly, thoroughly, and accurately based ONLY on the retrieved context below.

Formatting Guidelines:
- Explain the key concepts clearly with bullet points or numbered lists.
- If mathematical formulas (like Euclidean distance or centroid updates) are referenced, write them out clearly using standard LaTeX ($...$ or $$...$$).
- Include inline citations like [Source: <filename>, Page: <page>] for facts and equations.
- If the context does not contain the answer, reply ONLY with: "The provided documentation does not contain information to answer this question."

Retrieved Context:
{formatted_ctx}

Question: {standalone}
Answer:"""

        return standalone, docs, citations, qa_prompt

    def chat(self, session_id: str, message: str, strategy: str = "hybrid", timeline=None) -> Dict[str, Any]:
        """Non-streaming variant used by POST /api/rag/chat.

        Internally still uses generate_content_stream so we can log genuine
        time-to-first-token, even though the endpoint itself returns one
        final assembled answer rather than SSE tokens.
        """
        sid = session_id or "default_session"
        standalone, docs, citations, qa_prompt = self._retrieve_and_build_prompt(sid, message, strategy)

        answer = ""
        chunks_count = 0
        llm_error = True
        if self.client:
            for m in self.models_to_try:
                try:
                    if timeline:
                        timeline.mark("llm.request_sent", model=m, prompt_chars=len(qa_prompt))
                    stream = self.client.models.generate_content_stream(model=m, contents=qa_prompt)
                    pieces = []
                    got_first_chunk = False
                    for chunk in stream:
                        if chunk.text:
                            if not got_first_chunk:
                                got_first_chunk = True
                                if timeline:
                                    timeline.mark("llm.first_chunk_received", chars=len(chunk.text))
                            pieces.append(chunk.text)
                            chunks_count += 1
                    answer = "".join(pieces).strip()
                    llm_error = False
                    if timeline:
                        timeline.mark("llm.last_chunk_received", chunks=chunks_count, answer_chars=len(answer))
                    break
                except Exception as e:
                    print(f"[RAG Service] Model {m} failed: {e}")
                    continue

        if not answer:
            answer = "The provided documentation does not contain information to answer this question."

        if timeline:
            timeline.mark(
                "chat.answer_ready",
                chunks_retrieved=len(docs), answer_chars=len(answer), llm_error=llm_error,
            )

        history = self.get_history(sid)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        return {"answer": answer, "citations": citations, "standalone_query": standalone}

    def chat_stream(self, session_id: str, message: str, strategy: str = "hybrid", timeline=None) -> Generator[str, None, None]:
        """Streams response tokens using Server-Sent Events (SSE) protocol."""
        sid = session_id or "default_session"
        standalone, docs, citations, qa_prompt = self._retrieve_and_build_prompt(sid, message, strategy)
        request_id = timeline.request_id if timeline else None

        # 1. Yield initial metadata block
        metadata_payload = {
            "type": "metadata",
            "session_id": sid,
            "standalone_query": standalone,
            "strategy": strategy,
            "citations": citations,
            "request_id": request_id,
        }
        yield f"data: {json.dumps(metadata_payload)}\n\n"

        full_answer = []
        chunks_count = 0
        llm_error = True
        if self.client:
            for m in self.models_to_try:
                try:
                    if timeline:
                        timeline.mark("llm.request_sent", model=m, prompt_chars=len(qa_prompt))
                    stream = self.client.models.generate_content_stream(model=m, contents=qa_prompt)
                    got_first_chunk = False
                    for chunk in stream:
                        if chunk.text:
                            if not got_first_chunk:
                                got_first_chunk = True
                                if timeline:
                                    timeline.mark("llm.first_chunk_received", chars=len(chunk.text))
                            full_answer.append(chunk.text)
                            chunks_count += 1
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk.text})}\n\n"
                    llm_error = False
                    if timeline:
                        timeline.mark("llm.last_chunk_received", chunks=chunks_count, answer_chars=len("".join(full_answer)))
                    break
                except Exception as e:
                    print(f"[RAG Service] Stream model {m} failed: {e}")
                    continue

        complete_text = "".join(full_answer).strip()
        if not complete_text:
            complete_text = "The provided documentation does not contain information to answer this question."
            yield f"data: {json.dumps({'type': 'token', 'content': complete_text})}\n\n"

        if timeline:
            timeline.mark(
                "chat.answer_ready",
                chunks_retrieved=len(docs), answer_chars=len(complete_text), llm_error=llm_error,
            )

        # Update session history
        history = self.get_history(sid)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": complete_text})

        # Yield completion signal
        yield f"data: {json.dumps({'type': 'done', 'full_answer': complete_text, 'request_id': request_id})}\n\n"