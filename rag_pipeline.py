from typing import List
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from vector_store import VectorStore
import re

# --------------------------------------------------
# Context Limiting Configuration
# --------------------------------------------------
# Local LLMs like Ollama/Phi have strict context limits.
# Sending too much text causes crashes, hangs, or OOM errors.
# We limit context to ~6000 chars to stay well within safe bounds.
MAX_CONTEXT_CHARS = 6000

# Default retrieval reduced from 25-30 to 6-8 to prevent overload
DEFAULT_RETRIEVAL_K = 7

# Similarity filtering tuned for MiniLM + FAISS (cosine/L2). We convert
# FAISS distances into a monotonic similarity score in [0,1] and drop
# items that are too weak relative to the best match.
MIN_ABS_SIMILARITY = 0.25

# Cap how many chunks we allow from a single source to preserve diversity.
MAX_CHUNKS_PER_SOURCE = 3


# --------------------------------------------------
# Utility: Simple regex-based question extractor
# --------------------------------------------------
def extract_only_questions(text: str) -> str:
    lines = text.split("\n")
    questions = []

    for line in lines:
        line = line.strip()
        
        # Must end with ?
        if not line.endswith("?"):
            continue

        # Remove junk lines
        banned = ["answer", "import", "example", "code", "plt.", "library", "http"]
        if any(word in line.lower() for word in banned):
            continue

        # Remove option markers like a), b), c)
        line = re.sub(r"\b[a-d]\)", "", line).strip()

        # Length filter
        if len(line) < 10 or len(line) > 180:
            continue

        questions.append(line)

    # Remove duplicates
    questions = list(dict.fromkeys(questions))

    if not questions:
        return "No questions found in this document."

    return "\n".join([f"• {q}" for q in questions])


# --------------------------------------------------
# RAG Pipeline
# --------------------------------------------------
class RAGPipeline:
    def __init__(self, vector_store: VectorStore, model: str = "phi"):
        self.vector_store = vector_store
        self.llm = Ollama(model=model)

        # Prompt templates
        self.qa_template = self._create_qa_template()
        self.summary_template = self._create_summary_template()
        self.extract_template = self._create_extraction_template()

    # -------------------------------
    # Prompt Templates
    # -------------------------------
    def _create_qa_template(self):
        template = """
You are a helpful assistant that answers strictly using the provided context.

CONTEXT:
{context}

QUESTION:
{question}

RULES:
- Use only the provided context.
- If the answer is not present, say:
  "I don't have enough information in the documents."
- Do NOT hallucinate.

ANSWER:
"""
        return ChatPromptTemplate.from_template(template)

    def _create_summary_template(self):
        template = """
You are a helpful assistant.

TASK:
Summarize the following document content clearly and concisely.

RULES:
- Use only the provided context.
- Do NOT hallucinate.
- Keep it short and structured.
- Use bullet points if helpful.

DOCUMENT:
{context}

SUMMARY:
"""
        return ChatPromptTemplate.from_template(template)

    def _create_extraction_template(self):
        template = """
You are a STRICT question-extraction system.

RULES:
1. Extract ONLY main questions explicitly written in the document.
2. Remove sub-parts like a), b), 1), i), etc.
3. Ignore answers, examples, explanations.
4. Each output line must end with '?'.
5. Do NOT paraphrase.
6. Deduplicate questions.
7. If no valid questions exist, output exactly:
   "No questions found in this document."

FORMAT:
• Question?

DOCUMENT:
{context}

EXTRACTED QUESTIONS:
"""
        return ChatPromptTemplate.from_template(template)

    # -------------------------------
    # Retrieval Helpers
    # -------------------------------
    def _normalize_query(self, query: str) -> str:
        """Normalize queries before embedding to improve semantic precision.

        Lowercasing and whitespace collapsing makes intent-focused queries
        (e.g., "list all questions", "summarize this document") embed more
        consistently, reducing vector noise across uploads.
        """
        query = query.strip().lower()
        # Collapse repeated whitespace
        query = re.sub(r"\s+", " ", query)
        return query

    def _score_to_similarity(self, score: float) -> float:
        """Convert FAISS distance to a comparable similarity score in [0,1].

        FAISS returns smaller-is-better L2 distances for MiniLM embeddings.
        We invert with a stable transform so we can sort descending and apply
        intuitive thresholds without altering the underlying index behavior.
        """
        if score is None:
            return 0.0
        # 1 / (1 + d) is monotonic and keeps values in (0,1].
        return 1.0 / (1.0 + float(score))

    def _dedupe_and_limit(self, scored_docs, k: int):
        """Deduplicate near-identical chunks and limit per source for diversity."""
        seen_keys = set()
        per_source = {}
        kept = []

        for doc, sim in scored_docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page") or doc.metadata.get("page_number")
            
            # Early exit if per-source limit reached
            if per_source.get(src, 0) >= MAX_CHUNKS_PER_SOURCE:
                continue
            
            # Use a short text signature to catch repeated chunks from the same page
            text = doc.page_content or ""
            signature = text[:80].strip().lower()  # Reduced from 120 for faster hashing
            key = (src, page, signature)

            if key in seen_keys:
                continue

            seen_keys.add(key)
            per_source[src] = per_source.get(src, 0) + 1
            kept.append((doc, sim))

            if len(kept) >= k:
                break

        return kept

    # -------------------------------
    # Retrieval
    # -------------------------------
    def retrieve_context(self, query: str, k: int = None) -> List[Document]:
        """Retrieve high-quality context with filtering, rerank, and diversity.

        Steps applied (each keeps retrieval stable across multiple PDFs):
        - Normalize the query to cut embedding noise for intent-style asks.
        - Over-fetch then score-filter to remove weak/noisy matches.
        - Rerank by similarity (higher is better) and deduplicate repeated chunks.
        - Limit per source to preserve document diversity.
        """
        if k is None:
            k = DEFAULT_RETRIEVAL_K

        normalized_query = self._normalize_query(query)

        # Over-fetch to allow filtering without starving the final set.
        fetch_k = max(k * 2, k + 3)

        try:
            raw_results = self.vector_store.similarity_search_with_score(
                normalized_query,
                k=fetch_k,
            )
        except AttributeError:
            # Fallback for stores without score support; keeps API stable.
            raw_docs = self.vector_store.similarity_search(normalized_query, k=fetch_k)
            raw_results = [(doc, None) for doc in raw_docs]

        if not raw_results:
            return []

        # Convert to similarity (higher is better) and keep alongside docs.
        scored = []
        for doc, score in raw_results:
            similarity = self._score_to_similarity(score)
            # Store for observability; harmless to downstream consumers.
            doc.metadata = {**doc.metadata, "similarity": round(similarity, 4)}
            scored.append((doc, similarity))

        # Score-based filtering: drop items far below the best hit and under
        # an absolute floor to reduce unrelated chunks.
        best_sim = max(sim for _, sim in scored)
        dynamic_floor = max(MIN_ABS_SIMILARITY, best_sim - 0.25)
        scored = [(doc, sim) for doc, sim in scored if sim >= dynamic_floor]

        # Rerank high → low similarity for deterministic ordering.
        scored.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate repeated chunks and cap per source for diversity.
        scored = self._dedupe_and_limit(scored, k)

        # If filtering became too strict, fall back to the top-k raw order.
        if not scored:
            scored = raw_results[:k]
            scored = [(doc, self._score_to_similarity(score)) for doc, score in scored]

        # Final deterministic sort and trim.
        scored.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in scored[:k]]

        return final_docs

    # -------------------------------
    # Generators
    # -------------------------------
    def _truncate_context_safely(self, documents: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """Truncate combined document text to stay within safe context limits.
        
        Local models like Ollama crash when context is too large.
        This ensures we never exceed the safe character limit.
        """
        combined = []
        total_chars = 0
        
        for doc in documents:
            content = doc.page_content
            if total_chars + len(content) > max_chars:
                # Add partial content if space remains
                remaining = max_chars - total_chars
                if remaining > 100:  # Only add if meaningful space left
                    combined.append(content[:remaining] + "...")
                break
            combined.append(content)
            total_chars += len(content)
        
        return "\n\n".join(combined)

    def generate_response(self, query: str, documents: List[Document]) -> str:
        """Generate response with safe context limiting to prevent LLM crashes."""
        # Check if context is empty
        if not documents:
            return "I don't have enough information in the documents to answer this question."
        
        # Truncate context safely for local LLM stability
        context_text = self._truncate_context_safely(documents)

        prompt = self.qa_template.format(
            context=context_text,
            question=query
        )

        response = self.llm.invoke(prompt)
        
        # Extract clean text from response
        if hasattr(response, 'content'):
            return response.content.strip()
        else:
            return str(response).strip()

    def generate_summary(self, documents: List[Document]) -> str:
        """Generate summary with safe context limiting."""
        # Truncate context safely for local LLM stability
        context_text = self._truncate_context_safely(documents)

        prompt = self.summary_template.format(context=context_text)
        response = self.llm.invoke(prompt)

        return response

    def extract_questions_from_docs(self, documents: List[Document]) -> str:
        """Extract questions using per-document batching to prevent context overload.
        
        'List all questions' queries can retrieve many documents, causing
        Ollama to crash if sent all at once. We process each document
        separately and combine the results.
        """
        if not documents:
            return "No questions found in the documents."
        
        # Group documents by source to avoid redundant processing
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        all_questions = set()  # Use set to deduplicate across documents
        
        # Process each source document separately to stay within context limits
        for source, doc_list in docs_by_source.items():
            # Combine chunks from same source but respect context limit
            context_text = self._truncate_context_safely(doc_list, max_chars=4000)
            
            prompt = self.extract_template.format(context=context_text)
            
            try:
                response = self.llm.invoke(prompt)
                # Extract questions and add to set (deduplicates automatically)
                extracted = extract_only_questions(response)
                if "No questions found" not in extracted:
                    for line in extracted.split("\n"):
                        line = line.strip()
                        if line and line.startswith("•"):
                            all_questions.add(line)
            except Exception as e:
                # Continue processing other documents even if one fails
                print(f"Warning: Failed to extract from {source}: {e}")
                continue
        
        if not all_questions:
            return "No questions found in this document."
        
        # Return sorted list for consistent output
        return "\n".join(sorted(all_questions))

    # -------------------------------
    # Main Query Router
    # -------------------------------
    def query(self, question: str, k: int = None) -> dict:
        """Main query interface with intelligent routing and safe context handling.
        
        Args:
            question: User's query
            k: Number of documents to retrieve (default: 7 for safety)
        
        Returns:
            dict with mode, question, answer, and sources
        """
        # Use safe default if k not specified
        if k is None:
            k = DEFAULT_RETRIEVAL_K
        
        context = self.retrieve_context(question, k)

        q = question.lower()

        # ----------------------
        # Mode Detection
        # ----------------------
        # "List all questions" uses special batched processing to avoid crashes
        if "list" in q and "question" in q:
            mode = "extract"
            answer = self.extract_questions_from_docs(context)

        elif "summary" in q or "summarize" in q or "overview" in q:
            mode = "summary"
            answer = self.generate_summary(context)

        else:
            mode = "qa"
            answer = self.generate_response(question, context)

        return {
            "mode": mode,
            "question": question,
            "answer": answer,
            "sources": [doc.metadata for doc in context],
        }
