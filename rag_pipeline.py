from typing import List
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from vector_store import VectorStore


class RAGPipeline:
    def __init__(self, vector_store: VectorStore, model: str = "tinyllama"):
        self.vector_store = vector_store
        self.llm = Ollama(model=model)
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self):
        template = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

If the answer is not in the context, say:
"I don't have enough information to answer that question based on the provided documents."

Answer:"""

        return ChatPromptTemplate.from_template(template)

    def retrieve_context(self, query: str, k: int = 4) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def generate_response(self, query: str, context: List[Document]) -> str:
        context_text = "\n\n".join([doc.page_content for doc in context])

        prompt = self.prompt_template.format(
            context=context_text,
            question=query
        )

        response = self.llm.invoke(prompt)
        return response

    def query(self, question: str, k: int = 4) -> dict:
        context = self.retrieve_context(question, k)
        answer = self.generate_response(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": [doc.metadata for doc in context],
        }
