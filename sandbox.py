import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent

# --- SETUP ---
# Load API keys from .env file
load_dotenv()


@tool
def get_current_time():
    """
    Use this function to get the current date and time.
    Call this whenever a user asks for the time, the date, or anything related to the current moment.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


llm = ChatOpenAI(model="gpt-4o", temperature=0)

# The agent needs a list of all the tools it can use. For now, it's just one.
tools = [get_current_time]

# --- 3. CREATE THE AGENT ---
# The prompt is the agent's "brain". It tells the agent how to behave and how to use tools.
# Note the `agent_scratchpad` placeholder. This is where the agent does its "thinking".
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# `create_openai_tools_agent` creates the agent logic by binding the LLM to the tools and prompt.
agent = create_openai_tools_agent(llm, tools, prompt)

# --- 4. CREATE THE AGENT EXECUTOR ---
# The AgentExecutor is what actually runs the agent, calls the tools, and gets the results back.
# `verbose=True` is fantastic for learning, as it shows the agent's thought process.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 5. RUN THE CHATBOT ---
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # We invoke the agent executor with the user's input.
    # The input must be a dictionary.
    response = agent_executor.invoke({"input": user_input})

    # The final answer is in the "output" key of the response dictionary.
    print(f"Bot: {response['output']}")