import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


class VectorStore:
    def __init__(self, persist_directory: str = "./faiss_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None

    def create_vectorstore(self, chunks: List[Document]):
        # Create FAISS vector store from documents
        self.vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        # Persist locally
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vectorstore.save_local(self.persist_directory)

    def load_vectorstore(self):
        # Load existing FAISS vector store
        if not os.path.exists(self.persist_directory):
            raise ValueError("Vector store does not exist. Please create it first.")

        self.vectorstore = FAISS.load_local(
            self.persist_directory,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 8):
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore.similarity_search_with_score(query, k=k)
