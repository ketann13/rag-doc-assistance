import os
from pathlib import Path
from typing import List
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_documents(self, data_dir: str) -> List:
        documents = []
        data_path = Path(data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Directory {data_dir} does not exist")

        for file_path in data_path.iterdir():
            if file_path.is_file():
                if file_path.suffix == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
                elif file_path.suffix in ['.txt', '.md']:
                    loader = TextLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)

        if not documents:
            raise ValueError(f"No supported documents found in {data_dir}")

        return documents

    def split_documents(self, documents: List) -> List:
        chunks = self.text_splitter.split_documents(documents)
        return chunks
