
import os
import json
from datetime import datetime
import chainlit as cl

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# --- Load Configs ---
VECTOR_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "vector_config.json")
CHAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "chat_config.json")

with open(VECTOR_CONFIG_PATH, "r", encoding="utf-8") as f:
    vector_config = json.load(f)
with open(CHAT_CONFIG_PATH, "r", encoding="utf-8") as f:
    chat_config = json.load(f)

# Embedding & DB config
EMBED_MODEL_PATH = vector_config["EMBED_MODEL_PATH"]
EMBED_N_CTX = vector_config["EMBED_N_CTX"]
VECTOR_SIZE = vector_config["VECTOR_SIZE"]
DB_PATH = vector_config["DB_PATH"]

# Chat/LLM config
CHAT_LOG_DIR = os.path.join(os.getcwd(), "conversations")
COLLECTION_NAME = chat_config["COLLECTION_NAME"]
CHAT_MODEL_PATH = chat_config["CHAT_MODEL_PATH"]
LLM_TEMPERATURE = chat_config["LLM_TEMPERATURE"]
N_CTX_PER_SEQ = chat_config["N_CTX_PER_SEQ"]
RETRIEVAL_NUM_DOCS = chat_config["RETRIEVAL_NUM_DOCS"]
RETRIEVAL_SIMILARITY_THRESHOLD = chat_config["RETRIEVAL_SIMILARITY_THRESHOLD"]
SYSTEM_INSTRUCTIONS = chat_config["SYSTEM_INSTRUCTIONS"]
MAX_TOKENS = chat_config.get("MAX_TOKENS", 512)

os.makedirs(CHAT_LOG_DIR, exist_ok=True)

# --- Helper Functions ---

def get_langchain_llm():
    return LlamaCpp(
        model_path=CHAT_MODEL_PATH,
        n_ctx=N_CTX_PER_SEQ,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        verbose=False
    )

def get_embedding_model():
    return LlamaCppEmbeddings(
        model_path=EMBED_MODEL_PATH,
        n_ctx=EMBED_N_CTX,
        verbose=False
    )

def get_langchain_vectorstore():
    client = QdrantClient(path=DB_PATH)
    embedding_model = get_embedding_model()
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )

def get_memory(conversation_history):
    if not conversation_history:
        return None
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    for turn in conversation_history:
        user_msg = turn.get("user")
        assistant_msg = turn.get("assistant")
        if user_msg and assistant_msg:
            memory.save_context({"question": user_msg}, {"answer": assistant_msg})
    return memory

def format_chat_history(conversation_history):
    formatted = []
    for turn in conversation_history:
        if "user" in turn and "assistant" in turn:
            formatted.append(HumanMessage(content=turn["user"]))
            formatted.append(AIMessage(content=turn["assistant"]))
    return formatted

# --- Chainlit Events ---

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("conversation_history", [])
    cl.user_session.set("llm", get_langchain_llm())
    cl.user_session.set("vectorstore", get_langchain_vectorstore())
    cl.user_session.set("retriever", cl.user_session.get("vectorstore").as_retriever(
        search_kwargs={
            "k": RETRIEVAL_NUM_DOCS,
            "score_threshold": RETRIEVAL_SIMILARITY_THRESHOLD
        }
    ))
    cl.user_session.set("memory", get_memory(cl.user_session.get("conversation_history")))
    chain = ConversationalRetrievalChain.from_llm(
        llm=cl.user_session.get("llm"),
        retriever=cl.user_session.get("retriever"),
        memory=cl.user_session.get("memory"),
        return_source_documents=True,
        return_generated_question=True,
        output_key="answer",
        verbose=False
    )
    cl.user_session.set("chain", chain)
    cl.user_session.set("first_turn", True)
    cl.user_session.set("log_file", os.path.join(CHAT_LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"))
    await cl.Message(content="Chat assistant is ready. Ask your question!").send()

@cl.on_message
async def on_message(message: cl.Message):
    user_query = message.content
    conversation_history = cl.user_session.get("conversation_history")
    retriever = cl.user_session.get("retriever")
    retrieved_docs = retriever.get_relevant_documents(user_query)

    # await cl.Message(content=f"Retrieved {len(retrieved_docs)} documents. Waiting for the AI to process them and answer back...").send()

    system = SYSTEM_INSTRUCTIONS if cl.user_session.get("first_turn") else None
    cl.user_session.set("first_turn", False)

    formatted_history = format_chat_history(conversation_history)
    chain = cl.user_session.get("chain")
    result = await chain.ainvoke({"question": user_query, "chat_history": formatted_history})
    response = result["answer"]

    payload = {
        "status": "success",
        "error_message": "",
        "response": response
    }

    # await cl.Message(content=payload["response"]).send()
    await cl.Message(content=f"Retrieved {len(retrieved_docs)} documents. \n\n{response}").send()

    

    if system:
        conversation_history.append({"system": system, "user": user_query, "assistant": response})
    else:
        conversation_history.append({"user": user_query, "assistant": response})

    with open(cl.user_session.get("log_file"), "w", encoding="utf-8") as jf:
        json.dump(conversation_history, jf, ensure_ascii=False, indent=2)
