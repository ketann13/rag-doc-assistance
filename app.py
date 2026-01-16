import shutil
import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

DATA_DIR = "data"

st.set_page_config(page_title="Offline RAG Chatbot", layout="wide")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("Document Manager")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if st.sidebar.button("Save Documents"):
    if uploaded_files:
        os.makedirs(DATA_DIR, exist_ok=True)

        # Save files
        for file in uploaded_files:
            file_path = os.path.join(DATA_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        st.sidebar.success("Documents saved successfully!")

        #  Auto rebuild vector DB
        with st.spinner("Rebuilding vector database automatically..."):
            processor = DocumentProcessor()
            documents = processor.load_documents(DATA_DIR)
            chunks = processor.split_documents(documents)

            vector_store = VectorStore()
            vector_store.create_vectorstore(chunks)

            # Clear RAG pipeline cache only (not all resources)
            st.session_state.pop('rag_pipeline', None)

        st.sidebar.success("Vector database rebuilt automatically!")

    else:
        st.sidebar.warning("Please upload at least one document.")

# ------------------------
# Main UI
# ------------------------
st.title("Offline RAG Chatbot")
st.caption("Ask questions from your documents using local AI")


# Cache pipeline so it loads once
@st.cache_resource
def load_pipeline():
    vector_store = VectorStore()
    rag = RAGPipeline(vector_store)
    return rag

rag_pipeline = load_pipeline()

query = st.text_input("Ask your question:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            result = rag_pipeline.query(query, k=12)

        st.subheader("Answer")
        st.write(result["answer"])


        st.subheader("Sources")
        for idx, source in enumerate(result["sources"], 1):
            st.write(f"**Source {idx}:** {source}")
