from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ollama import chat


class RAGPipeline:

    def __init__(self, pdf_path,
                 chunk_size,
                 chunk_overlap,
                 top_k):

        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    def build_vector_db(self):

        loader = PyPDFLoader(self.pdf_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(
            chunks,
            embeddings
        )

        return db

    def retrieve_context(self, question):

        db = self.build_vector_db()

        docs = db.similarity_search(
            question,
            k=self.top_k
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

        return context

    def answer(self, question):

        context = self.retrieve_context(question)

        prompt = f"""
You are a helpful assistant.

Context:
{context}

Question:
{question}

Answer only from the context.
"""

        response = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]