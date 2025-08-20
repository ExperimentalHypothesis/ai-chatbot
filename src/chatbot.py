import warnings

from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.prompts import USER_REPHRASE_PROMPT

from src.config import Settings
from src.embedding_service import EmbeddingService
from src.prompts import QA_SYSTEM_PROMPT


class Chatbot:
    """
    A stateful chatbot that orchestrates the RAG chain and answers questions in a conversational style.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding = EmbeddingService(settings) # TODO: dep inj. (?)
        self.retriever = self.embedding.get_retriever()

        # init all the crap needed for full-conversational mode
        self.llm = self._init_llm()
        self.memory = self._init_memory()
        self.history_aware_retriever = self._init_history_aware_retriever()
        self.document_combiner = self._init_document_combiner()
        self.rag_chain = self._init_rag_chain()

    def ask(self, question: str) -> str:
        chat_memory = self.memory.load_memory_variables({})[self.settings.CHAT_MEMORY_KEY]
        response = self.rag_chain.invoke({"input": question, self.settings.CHAT_MEMORY_KEY: chat_memory})
        self.memory.save_context({"input": question}, {"output": response["answer"]})

        answer = response["answer"]
        source_docs = response.get("context", [])

        if source_docs:
            sources_list = []
            for doc in source_docs:
                source_name = doc.metadata.get('source', 'Unknown Source')
                sources_list.append(f"- {source_name}")
            if sources_list:
                answer += "\n\nSources:\n" + "\n".join(sorted(list(set(sources_list))))

        return answer

    def clear_chat_history(self):
        self.memory.clear()
        print("Chat history cleared.")

    def _init_memory(self):
        """
        Memory for multiturn conversation.
        """
        return ConversationBufferWindowMemory(
            k=self.settings.CHAT_TURNS,
            # Stores the last 5 conversation turns (user input + AI response), total 10 messages
            memory_key=self.settings.CHAT_MEMORY_KEY,  # The key used in the prompt templates for history
            return_messages=True,  # Return messages as LangChain message objects
            ai_prefix="Assistant"  # Label for AI messages in history
        )

    def _init_history_aware_retriever(self):
        """
        History-aware Retriever Chain (for Multi-Turn Conversations):
        This chain's job is to take the current user's question AND the chat history,
        then ask the LLM to rephrase the question into a standalone query suitable
        for semantic search. This handles references like "it" or "that".
        """
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name=self.settings.CHAT_MEMORY_KEY),  # Inject chat history here
            ("user", "{input}"),  # The current user's question
            ("user", USER_REPHRASE_PROMPT),
        ])
        return create_history_aware_retriever(
            self.llm,  # The LLM used to rephrase the query
            self.retriever,  # The underlying retriever that will get documents based on the rephrased query
            prompt  # The specific prompt to guide the rephrasing LLM
        )

    def _init_document_combiner(self):
        """
        Document Combination Chain (The core Q&A "stuffing" logic):
        This part takes the documents retrieved by the history-aware retriever
        and "stuffs" them together with the original user question (and history)
        into a single prompt for the LLM to generate the final answer.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT + "Context\n{context}"),
            MessagesPlaceholder(variable_name=self.settings.CHAT_MEMORY_KEY),
            ("user", "{input}"),
        ])
        return create_stuff_documents_chain(self.llm, prompt)

    def _init_rag_chain(self):
        """
        Final Retrieval Chain (The full RAG Pipeline Orchestrator):
        This orchestrates the entire RAG process:
        - First, it uses `history_aware_retriever` to get relevant documents (aware of chat history).
        - Then, it passes those documents along with the original user's question and chat history
        to `document_combiner` to generate the final answer from the LLM.
        """
        return create_retrieval_chain(
            self.history_aware_retriever,  # Handles getting relevant docs, aware of history
            self.document_combiner  # Handles combining docs + prompt for final answer
        )

    def _init_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.settings.LLM_MODEL,
            temperature=self.settings.LLM_TEMPERATURE,
            api_key=self.settings.OPENAI_API_KEY
        )

