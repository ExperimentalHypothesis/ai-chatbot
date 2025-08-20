import os
import logging
import gradio as gr

from src.chatbot import Chatbot
from src.config import settings
from src.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_embeddings():
    db_dir = settings.DB_DIR
    logger.info(f"Checking vector store '{db_dir}'...")

    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        logger.info(f"Vector store '{db_dir}' not found or empty. Processing documents...")
        service = EmbeddingService(settings)
        service.embed_documents()
    else:
        logger.info(f"Vector store '{db_dir}' found. Ready to use existing embeddings.")


def init_chatbot() -> Chatbot:
    """Initializes and returns the Chatbot instance."""
    try:
        chatbot = Chatbot(settings)
        return chatbot
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred during chatbot initialization: {e}")
        raise gr.Error(f"Failed to initialize the chatbot. Please check logs. Error: {e}")


def main():
    init_embeddings()
    chatbot = init_chatbot()
    def chat_function(message, history):
        logger.info(f"User query: {message}")
        response = chatbot.ask(message)
        logger.info(f"Chatbot response: {response}")
        return response

    logger.info("Launching Gradio Chat Interface...")
    ui = gr.ChatInterface(
        fn=chat_function,
        title="OpenModelica Chatbot",
        description="Ask questions, the chatbot will try to find answers.",
        theme="soft",
        submit_btn="Ask",
    )
    ui.launch()


if __name__ == "__main__":
    main()