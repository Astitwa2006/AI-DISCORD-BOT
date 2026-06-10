# AI Discord Bot

A conversational Discord bot built with Python, LangChain, and the OpenAI API. This bot is capable of answering questions about specific topics using **Retrieval-Augmented Generation (RAG)** based on a custom knowledge base, while also featuring real-time automated moderation and ultra-fast response caching.

## Features

- **Retrieval-Augmented Generation (RAG)**: The bot reads custom documents from a local `knowledge_base` directory, converts them into embeddings using Chroma, and retrieves relevant context to accurately answer user questions.
- **Automated Real-Time Moderation**: Integrates OpenAI's Moderation API to instantly analyze incoming messages. It automatically flags, deletes, and warns users about messages containing toxic behavior, hate speech, or harassment.
- **Lightning Fast Context Memory**: Utilizes LangChain's `SQLChatMessageHistory` backed by a local SQLite database to cache conversation context per user. This ensures the bot remembers previous interactions with a system response time optimized to under 1.5 seconds.

## Technologies Used
- **Python** & `discord.py`
- **LangChain** (for RAG chains and LCEL)
- **OpenAI API** (Embeddings, Chat models, Moderation)
- **SQLite** & `SQLAlchemy` (for local caching)
- **ChromaDB** (for local vector storage)

## Prerequisites
- Python 3.9+
- A Discord Bot Token (from the [Discord Developer Portal](https://discord.com/developers/applications))
- An OpenAI API Key

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Astitwa2006/AI-DISCORD-BOT.git
   cd AI-DISCORD-BOT
   ```

2. **Set up environment variables:**
   Rename `.env.example` to `.env` and insert your API keys:
   ```env
   DISCORD_TOKEN=your_discord_token_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

4. **Add your custom knowledge:**
   Drop any `.txt` files containing your domain-specific information into the `knowledge_base/` folder. The bot will automatically index these files upon startup.

5. **Run the bot:**
   ```bash
   python bot.py
   ```

## How It Works

1. **Startup**: When the bot is launched, it loads all `.txt` documents from the `knowledge_base`, chunks them, and creates a local Chroma vector database.
2. **Moderation Check**: Whenever a user sends a message, it is first routed through the OpenAI Moderation API. If the content is flagged as unsafe, it is instantly deleted and a warning is sent.
3. **RAG Processing**: Safe messages are passed to the LangChain RAG pipeline. The system retrieves conversation history from the SQLite database and relevant document snippets from ChromaDB to formulate a context-aware response using GPT-3.5-turbo.
