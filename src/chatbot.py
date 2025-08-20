import warnings
from datetime import datetime
from functools import partial

from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.prompts import USER_REPHRASE_PROMPT, AGENT_SYSTEM_PROMPT

from src.config import Settings
from src.embedding_service import EmbeddingService
from src.prompts import QA_SYSTEM_PROMPT

from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent

from src.tools import get_current_time, answer_questions_from_documents, save_conversation


class Chatbot:
    """
    A stateful chatbot that orchestrates the RAG chain and answers questions in a conversational style.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding = EmbeddingService(settings)
        self.retriever = self.embedding.get_retriever()

        # init all needed for full-conversational mode
        self.llm = self._init_llm()
        self.memory = self._init_memory()
        self.history_aware_retriever = self._init_history_aware_retriever()
        self.document_combiner = self._init_document_combiner()
        self.rag_chain = self._init_rag_chain()

        # init agent needed for tool picking
        self.agent_executor = self._init_agent_executor()

    def ask(self, question: str) -> str:
        """
        Main exposure point.
        """
        chat_memory = self.memory.load_memory_variables({})[self.settings.CHAT_MEMORY_KEY]

        response = self.agent_executor.invoke({
            "input": question,
            self.settings.CHAT_MEMORY_KEY: chat_memory
        })

        answer = response["output"]
        self.memory.save_context({"input": question}, {"output": answer})
        return answer

    def clear_chat_history(self):
        self.memory.clear()
        print("Chat history cleared.")

    @property
    def tools(self):
        """
        This method defines all the tools the agent can use.
        """
        # qa_tool = partial(answer_questions_from_documents, rag_chain=self.rag_chain, memory=self.memory)
        # qa_tool.__doc__ = answer_questions_from_documents.__doc__
        #
        # save_tool = partial(save_conversation, memory=self.memory)
        # save_tool.__doc__ = save_conversation.__doc__
        #
        # time_tool = get_current_time



        return [get_current_time]

    def _init_agent_executor(self):
        """
        Initializes the agent that can choose between tools.
        """
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name=self.settings.CHAT_MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(self.llm, self.tools, agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

        return agent_executor

    def _init_memory(self):
        """
        Memory for multiturn conversation.
        """
        return ConversationBufferWindowMemory(
            k=self.settings.CHAT_TURNS,
            memory_key=self.settings.CHAT_MEMORY_KEY,
            return_messages=True,
            ai_prefix="Assistant"
        )

    def _init_history_aware_retriever(self):
        """
        History-aware Retriever Chain (for Multi-Turn Conversations):
        This chain's job is to take the current user's question AND the chat history,
        then ask the LLM to rephrase the question into a standalone query suitable
        for semantic search. This handles references like "it" or "that".
        """
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name=self.settings.CHAT_MEMORY_KEY),
            ("user", "{input}"),  # The current user's question
            ("user", USER_REPHRASE_PROMPT),
        ])
        return create_history_aware_retriever(
            self.llm,
            self.retriever,
            prompt
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

