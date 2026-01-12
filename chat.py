import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


def initialize_rag(data_dir: str = "./data", vector_db_dir: str = "./chroma_db"):
    load_dotenv()

    if not os.path.exists(vector_db_dir):
        print("Creating new vector database from documents...")
        
        processor = DocumentProcessor()
        documents = processor.load_documents(data_dir)
        print(f"Loaded {len(documents)} document pages")

        chunks = processor.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        vector_store = VectorStore(vector_db_dir)
        vector_store.create_vectorstore(chunks)
        print("Vector database created successfully")
    else:
        print("Loading existing vector database...")
        vector_store = VectorStore(vector_db_dir)
        vector_store.load_vectorstore()
        print("Vector database loaded successfully")

    rag = RAGPipeline(vector_store)
    return rag


def chat_interface(rag: RAGPipeline):
    print("\n" + "="*50)
    print("RAG Chatbot - Ask questions about your documents")
    print("="*50)
    print("Type 'quit' or 'exit' to end the conversation\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not question:
                continue

            print("\nSearching documents...")
            result = rag.query(question)

            print(f"\nBot: {result['answer']}")

            if result['sources']:
                print(f"\nSources used: {len(result['sources'])} documents")
            
            print("\n" + "-"*50 + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


def main():
    try:
        rag = initialize_rag()
        chat_interface(rag)
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        print("Make sure you have:")
        print("1. Added documents to the 'data' directory")
        print("2. Set your OPENAI_API_KEY in the .env file")


if __name__ == "__main__":
    main()
