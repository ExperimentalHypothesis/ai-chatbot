
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
            unique_sources = set()
            for doc in source_docs:
                source = doc.metadata.get('source', 'Unknown Source')
                page = doc.metadata.get('page', '')
                page_info = f"page {page + 1}" if page != '' else "Unknown Page"
                unique_sources.add(f"- {source} ({page_info})")

            if unique_sources:
                sources_text = "\n".join(sorted(list(unique_sources)))
                answer += f"\n\n**Sources:**\n{sources_text}"
        return answer

    return qa_documents

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
            f.write(conversation_history)

        return f"Conversation successfully saved to '{filename}'."
    except Exception as e:
        return f"Error: Failed to save conversation. {e}"

