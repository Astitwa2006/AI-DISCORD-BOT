import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from context_cache import get_message_history

BASE_DIR = os.path.dirname(__file__)
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# Global variables for caching the chain
_qa_chain_with_history = None

def initialize_rag():
    """
    Initializes the RAG system by loading documents, creating embeddings, 
    setting up the vector store, and creating the conversational chain.
    """
    global _qa_chain_with_history
    if _qa_chain_with_history is not None:
        return _qa_chain_with_history

    # 1. Load documents
    if not os.path.exists(KB_DIR):
        os.makedirs(KB_DIR)
        
    loader = DirectoryLoader(KB_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # If no documents are found, we'll just use a fallback or empty list.
    # We still need a valid retriever.
    if not documents:
        print("Warning: No documents found in the knowledge base directory. RAG might not have context.")
        # Fallback empty document so Chroma doesn't fail
        from langchain_core.documents import Document
        documents = [Document(page_content="I am an AI assistant.", metadata={"source": "system"})]

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 3. Create Vector Store and Retriever
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Create the LLM and Prompt
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    from langchain.chains import create_history_aware_retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "You are a helpful assistant for a Discord community.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer concise and suitable for a chat platform like Discord.\n\n"
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 5. Add Message History (SQLite backed)
    _qa_chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return _qa_chain_with_history

def get_answer(user_id: str, question: str) -> str:
    """
    Given a user ID (used as session ID for memory) and a question,
    returns the AI's response using RAG.
    """
    chain = initialize_rag()
    
    response = chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": str(user_id)}}
    )
    
    return response["answer"]
