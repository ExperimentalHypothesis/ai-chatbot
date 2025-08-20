import os
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from src.config import Settings

logging.basicConfig(level=logging.INFO)

class EmbeddingService:
    """
    Service class for embedding and vector retrieval.
    Responsible for data preparation pipeline loading, splitting, embedding.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.embedding_model = self._init_embedding_model()
        self.text_splitter = self._init_text_splitter()
        self.vector_store = None

    def get_retriever(self, k: int):
        if self.vector_store is None:
            self.logger.info(f"Attempting to load vector store {self.settings.DB_DIR}")

            try:
                self.vector_store = Chroma(
                    persist_directory=self.settings.DB_DIR,
                    embedding_function=self.embedding_model,
                    collection_name=self.settings.DB_COLLECTION
                )
                self.logger.info("Vector store loaded successfully.")
            except Exception as e:
                self.logger.error(f"Could not load vector store from '{self.settings.DB_DIR}', collection '{self.settings.DB_COLLECTION}': {e}")
                self.logger.error("It might not exist or is corrupted. Please try running `embed_documents()` first.")
                return None
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def embed_documents(self):
        self.logger.info(f"Embedding documents into collection: '{self.settings.DB_COLLECTION}'...")

        documents = self._load_pdf()
        if not documents:
            self.logger.warning(f"No documents found to embed. Please check your directory.")
            return

        self.logger.info(f"Splitting {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store = Chroma.from_documents(
            collection_name=self.settings.DB_COLLECTION,
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.settings.DB_DIR,
        )
        self.logger.info(f"Documents processed and stored successfully in {self.settings.DB_DIR}.")

    def _load_pdf(self):
        self.logger.info("Loading documents...")

        all_documents = []
        file_path = os.path.join(self.settings.DOCS_DIR, self.settings.DOCS_FILENAME)

        if not os.path.exists(file_path):
            self.logger.warning(f"Document path '{file_path}' does not exist.")
            return []

        loader = PyPDFLoader(file_path)
        loaded_docs = loader.load()
        all_documents.extend(loaded_docs)
        return all_documents


    def _init_embedding_model(self) -> Embeddings:
        self.logger.info("Initializing embedding model...")

        return OpenAIEmbeddings(
            model=self.settings.LLM_EMBEDDING_MODEL,
            api_key=self.settings.OPENAI_API_KEY
        )

    def _init_text_splitter(self):
        self.logger.info("Initializing text splitter..")

        return RecursiveCharacterTextSplitter(
            chunk_size=self.settings.SPLITTER_CHUNK_SIZE,
            chunk_overlap=self.settings.SPLITTER_OVERLAP,
            length_function=len,
        )