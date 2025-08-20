import os

from src.chatbot import Chatbot
from src.config import settings
from src.embeddings import EmbeddingService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_embeddings():
    db_dir = settings.DB_DIR
    logger.info(f"Checking vector store '{db_dir}..'")

    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        logger.info(f"Vector store '{db_dir}' not found or empty. Processing documents...")
        service = EmbeddingService(settings)
        service.embed_documents()
        return service
    else:
        logger.info(f"Vector store '{db_dir}' found. Ready to use existing embeddings.")
        return None


def init_chatbot():
    try:
        chatbot = Chatbot(settings)
        return chatbot
    except Exception as e:
        logger.info(f"\nAn unexpected error occurred during chatbot initialization: {e}")
        return None


def chat_loop(chatbot):
    """Run the interactive chat loop."""
    logger.info("Chatbot initialized. Type 'clear' to reset chat, 'exit' to quit.")

    while True:
        user_input = input("\nYour question: ").strip()

        if user_input.lower() == 'exit':
            logger.info("Exiting...")
            break
        elif user_input.lower() == 'clear':
            chatbot.clear_chat_history()
            continue
        elif not user_input:
            continue

        try:
            logger.info("\nThinking...")
            response = chatbot.ask(user_input)
            logger.info(f"\nAnswer:\n{response}")
        except Exception as e:
            logger.info(f"An error occurred while processing your question: {e}")
            logger.info("Please try again or check your connection.")


def main():
    init_embeddings()
    chatbot = init_chatbot()
    chat_loop(chatbot)


if __name__ == "__main__":
    main()