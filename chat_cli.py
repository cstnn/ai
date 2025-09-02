# chat.py
# -------
# Interactive RAG chat assistant using local GGUF LLM and Qdrant DB for semantic retrieval.

# --- Imports ---
import os
import sys
import json
from datetime import datetime

# Qdrant client for vector database operations
from qdrant_client import QdrantClient

# Langchain integrations for Qdrant and LlamaCpp
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Qdrant as LCQdrant

from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import LlamaCppEmbeddings

# --- Load Configuration Files ---
print("### LOADING CONFIGS ###")
VECTOR_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "vector_config.json")
CHAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "chat_config.json")

# Load vector and chat configurations
with open(VECTOR_CONFIG_PATH, "r", encoding="utf-8") as f:
    vector_config = json.load(f)
with open(CHAT_CONFIG_PATH, "r", encoding="utf-8") as f:
    chat_config = json.load(f)

# Extract embedding and DB settings
EMBED_MODEL_PATH = vector_config["EMBED_MODEL_PATH"]
EMBED_N_CTX = vector_config["EMBED_N_CTX"]
VECTOR_SIZE = vector_config["VECTOR_SIZE"]
DB_PATH = vector_config["DB_PATH"]

# Extract chat/LLM settings
CHAT_LOG_DIR = os.path.join(os.getcwd(), "conversations")
COLLECTION_NAME = chat_config["COLLECTION_NAME"]
CHAT_MODEL_PATH = chat_config["CHAT_MODEL_PATH"]
LLM_TEMPERATURE = chat_config["LLM_TEMPERATURE"]
N_CTX_PER_SEQ = chat_config["N_CTX_PER_SEQ"]
RETRIEVAL_NUM_DOCS = chat_config["RETRIEVAL_NUM_DOCS"]
RETRIEVAL_SIMILARITY_THRESHOLD = chat_config["RETRIEVAL_SIMILARITY_THRESHOLD"]
SYSTEM_INSTRUCTIONS = chat_config["SYSTEM_INSTRUCTIONS"]
MAX_TOKENS = chat_config.get("MAX_TOKENS", 512)

# --- Create Directory for Chat Logs ---
print("### CREATING DIRECTORIES ###")
os.makedirs(CHAT_LOG_DIR, exist_ok=True)
LOG_BASENAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
JSON_FILE = os.path.join(CHAT_LOG_DIR, f"{LOG_BASENAME}.json")

# --- Model and Retrieval Setup Functions ---

def get_langchain_llm():
    """Initialize and return the LlamaCpp LLM with streaming enabled."""
    return LlamaCpp(
        model_path=CHAT_MODEL_PATH,
        n_ctx=N_CTX_PER_SEQ,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        streaming=True,
        verbose=False
    )

def get_embedding_model():
    """Initialize and return the embedding model using LlamaCpp."""
    return LlamaCppEmbeddings(
        model_path=EMBED_MODEL_PATH,
        n_ctx=EMBED_N_CTX,
        verbose=False
    )

def get_langchain_vectorstore():
    """Initialize Qdrant vector store with embedding model."""
    client = QdrantClient(path=DB_PATH)
    embedding_model = get_embedding_model()
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model
    )

class ScoredQdrantRetriever(Qdrant):
    """Custom retriever that includes similarity score in metadata."""
    def get_relevant_documents(self, query: str) -> list[Document]:
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=self._embedding.embed_query(query),
            limit=self.search_kwargs.get("k", 5),
            score_threshold=self.search_kwargs.get("score_threshold", None),
            with_payload=True,
            with_vectors=False
        )
        docs = []
        for result in results:
            metadata = result.payload or {}
            metadata["score"] = result.score
            docs.append(Document(page_content=metadata.get("page_content", ""), metadata=metadata))
        return docs

def get_memory(conversation_history):
    """Reconstruct memory from previous conversation history."""
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

def get_conversational_chain(llm, retriever, memory):
    """Create and return a ConversationalRetrievalChain."""
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
        output_key="answer",
        verbose=False
    )

def format_chat_history(conversation_history):
    """Convert conversation history into Langchain message format."""
    formatted = []
    for turn in conversation_history:
        if "user" in turn and "assistant" in turn:
            formatted.append(HumanMessage(content=turn["user"]))
            formatted.append(AIMessage(content=turn["assistant"]))
    return formatted

# --- Main Chat Loop ---

def main():
    print("### STARRTING CHAT LOOP ###")
    llm = get_langchain_llm()
    is_new_session = not os.path.exists(JSON_FILE)
    conversation_history = []

    vectorstore = get_langchain_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_NUM_DOCS,
            "score_threshold": RETRIEVAL_SIMILARITY_THRESHOLD
        }
    )

    retriever.search_kwargs = {
        "k": RETRIEVAL_NUM_DOCS,
        "score_threshold": RETRIEVAL_SIMILARITY_THRESHOLD
    }

    memory = get_memory(conversation_history)
    chain = get_conversational_chain(llm, retriever, memory)

    print("### Chat assistant is ready. Type your question !")

    if not is_new_session:
        with open(JSON_FILE, "r", encoding="utf-8") as jf:
            conversation_history = json.load(jf)

    first_turn = True
    while True:
        user_query = input("> User: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        retrieved_docs = retriever.get_relevant_documents(user_query)
        print(f"### Retrieved {len(retrieved_docs)} documents.")

        system = SYSTEM_INSTRUCTIONS if first_turn else None
        first_turn = False

        formatted_history = format_chat_history(conversation_history)
        result = chain.invoke({"question": user_query, "chat_history": formatted_history})
        response = result["answer"]
        print(f"> Assistant: {response}")

        if system:
            conversation_history.append({"system": system, "user": user_query, "assistant": response})
        else:
            conversation_history.append({"user": user_query, "assistant": response})

        with open(JSON_FILE, "w", encoding="utf-8") as jf:
            json.dump(conversation_history, jf, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
