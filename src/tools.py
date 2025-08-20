
from datetime import datetime
from langchain_core.tools import tool


def create_qa_tool(rag_chain, memory, memory_key):
    @tool
    def qa_documents(question: str) -> str:
        """
        Use this tool to answer user questions about the provided context.
        This should be your default tool for any informational query.
        """
        chat_memory = memory.load_memory_variables({})[memory_key]
        response = rag_chain.invoke(
            {"input": question, memory_key: chat_memory}
        )
        answer = response["answer"]
        source_docs = response.get("context", [])

        if source_docs:
            sources_list = [
                f"- {doc.metadata.get('source', 'Unknown Source')}" for doc in source_docs
            ]
            if sources_list:
                answer += "\n\nSources:\n" + "\n".join(sorted(list(set(sources_list))))
        return answer

    return qa_documents


@tool
def get_current_time():
    """
    Get the current date and time.
    Call this whenever a user asks for the time, the date, or anything related to the current moment.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def save_conversation(conversation_history: str) -> str:
    """
    Saves the provided conversation history string to a text file.
    Use this when the user explicitly asks to save, export, or write down the chat.
    The agent should provide the full conversation history as the 'conversation_history' argument.
    """
    if not conversation_history or not isinstance(conversation_history, str):
        return "Error: No valid conversation history was provided to save."

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write("Conversation History\n")
            f.write("=" * 20 + "\n\n")
            # The agent already formatted the history, so we just write it.
            f.write(conversation_history)

        return f"Conversation successfully saved to '{filename}'."
    except Exception as e:
        return f"Error: Failed to save conversation. {e}"

