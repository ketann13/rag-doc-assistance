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
    # Retrieval
    # -------------------------------
    def retrieve_context(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents with safe default k value."""
        if k is None:
            k = DEFAULT_RETRIEVAL_K
        return self.vector_store.similarity_search(query, k=k)

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

        # Balance chunks per document to avoid overloading from single source
        MAX_PER_DOC = 6
        filtered = []
        doc_counter = {}

        for doc in context:
            src = doc.metadata.get("source", "unknown")
            doc_counter[src] = doc_counter.get(src, 0) + 1

            if doc_counter[src] <= MAX_PER_DOC:
                filtered.append(doc)

        q = question.lower()

        # ----------------------
        # Mode Detection
        # ----------------------
        # "List all questions" uses special batched processing to avoid crashes
        if "list" in q and "question" in q:
            mode = "extract"
            answer = self.extract_questions_from_docs(filtered)

        elif "summary" in q or "summarize" in q or "overview" in q:
            mode = "summary"
            answer = self.generate_summary(filtered)

        else:
            mode = "qa"
            answer = self.generate_response(question, filtered)

        return {
            "mode": mode,
            "question": question,
            "answer": answer,
            "sources": [doc.metadata for doc in filtered],
        }
