# AI Chatbot for OpenModelica Documentation

Chatbot prototype designed to answer OpenModelica guide. It utilizes a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, context-aware answers from loaded document.

## Getting Started

Follow these instructions to set up and run the chatbot on your local machine.

### Prerequisites

- Python 3.13 or higher
- UV as package manager
- An OpenAI API key
- LangSmith api key for monitoring

### Installation

```
git clone 
cd ai-chatbot-openmodelica
python -m venv .venv 
source .venv/bin/activate 
uv sync
```

### Configuration

1.  **Create a `.env` file** in the root directory of the project.
```
OPENAI_API_KEY="sk..."
```

2.  **Run the Chatbot:**
Execute the `main.py` script from the root directory:
```bash
python main.py
```
- It will launch a GUI http://127.0.0.1:7860/  where you can type questions.

## How it works

- The first time you run the application, it will check if a vector store exists in the `db` directory.
- If not found, it will automatically process the document in the `docs` directory, create the embeddings, and save them to the `db` directory. It may take some time.
- Embedding it done only once, additional re-launch will skip it. 
- Once the vector store is ready, the interactive chat session will begin.

## Features

- **Retrieval-Augmented Generation (RAG):**  Answers grounded in the content of the documents only. It is actually very strict about that.
- **Persistent Vector Storage:** Uses ChromaDB to store document embeddings, so you only need to process your documents once.
- **Context-Aware Conversations:** Maintains a chat history to understand follow-up questions and provide coherent conversational answers.
- **Save Conversation:** Able to save conversation history a local .txt file 
- **Simple GUI:** A simple and easy-to-use GUI interface for interacting with the chatbot.


## TODOs: Next Steps & Improvements
- **Better chunking strategy:** This would have a major impact. There really should be a custom class that chunks the document into more logical parts (chapters/subchapters) then what `RecursiveCharacterTextSplitter` does.
- **Better storage:** 
- **Add some evaluation:**



