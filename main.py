import os

from src.chatbot import Chatbot
from src.config import settings
from src.embedding_service import EmbeddingService


def init_embeddings():
    db_dir = settings.DB_DIR
    print(f"Checking vector store '{db_dir}..'")

    if not os.path.exists(db_dir) or not os.listdir(db_dir):
        print(f"Vector store '{db_dir}' not found or empty. Processing documents...")
        service = EmbeddingService(settings)
        service.embed_documents()
        return service
    else:
        print(f"Vector store '{db_dir}' found. Ready to use existing embeddings.")
        return None


def init_chatbot():
    try:
        chatbot = Chatbot(settings)
        return chatbot
    except Exception as e:
        print(f"\nAn unexpected error occurred during chatbot initialization: {e}")
        return None


def chat_loop(chatbot):
    """Run the interactive chat loop."""
    print("\nChatbot initialized. Type 'clear' to reset chat, 'exit' to quit.")

    while True:
        user_input = input("\nYour question: ").strip()

        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        elif user_input.lower() == 'clear':
            chatbot.clear_chat_history()
            continue
        elif not user_input:
            continue

        try:
            print("\nThinking...")
            response = chatbot.ask(user_input)
            print(f"\nAnswer:\n{response}")
        except Exception as e:
            print(f"An error occurred while processing your question: {e}")
            print("Please try again or check your connection.")


def main():
    init_embeddings()
    chatbot = init_chatbot()
    chat_loop(chatbot)


if __name__ == "__main__":
    main()