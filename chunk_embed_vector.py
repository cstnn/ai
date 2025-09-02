
import os
import glob
from datetime import datetime
import uuid
import json
from qdrant_client.http.models import PointStruct
import qdrant_client
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


"""
chunk_embed_vector.py
---------------------
Document ingestion, chunking, embedding, and upsert to Qdrant vector DB for RAG applications.

Configuration is loaded from vector_config.json, which controls paths, chunking, model, and DB parameters.

Pipeline steps:
1. Load documents from DOC_DIR (supports .md, .pdf, .txt)
2. Chunk documents using RecursiveCharacterTextSplitter
3. Embed chunks using a local GGUF embedding model via llama-cpp-python
4. Upsert embeddings and metadata to Qdrant DB
5. Log all steps to EMBEDDINGS_LOG_DIR

Edit vector_config.json to change model, chunking, or DB settings.
"""

# --- Load Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "vector_config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
	config = json.load(f)

EMBEDDINGS_LOG_DIR = config["EMBEDDINGS_LOG_DIR"]  # Directory for embedding logs
DOC_DIR = config["DOC_DIR"]  # Directory containing source documents
CHUNK_SIZE = config["CHUNK_SIZE"]  # Chunk size for text splitting
CHUNK_OVERLAP = config["CHUNK_OVERLAP"]  # Overlap between chunks
EMBED_MODEL_PATH = config["EMBED_MODEL_PATH"]  # Path to GGUF embedding model
EMBED_N_CTX = config["EMBED_N_CTX"]  # Context window for embedding model
DB_PATH = config["DB_PATH"]  # Path to Qdrant DB
VECTOR_SIZE = config["VECTOR_SIZE"]  # Embedding vector size
COLLECTION_NAME = config["COLLECTION_NAME"]  # Qdrant collection name


# --- Logging Setup ---
os.makedirs('models', exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(EMBEDDINGS_LOG_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))

def log_event(level, step, text):
	"""Log a pipeline event to the log file."""
	ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	log_line = f"{ts} - {level} - {step} - {text}\n"
	with open(LOG_FILE, "a", encoding="utf-8") as f:
		f.write(log_line)

# --- Document Loading ---
DOC_DIR = DOC_DIR
def load_documents():
	"""Load documents from DOC_DIR using supported loaders."""
	docs = []
	for ext, loader_cls in [("*.md", UnstructuredMarkdownLoader), ("*.pdf", PDFPlumberLoader), ("*.txt", TextLoader)]:
		for path in glob.glob(os.path.join(DOC_DIR, ext)):
			try:
				loader = loader_cls(path)
				loaded = loader.load()
				docs.extend(loaded)
				log_event("INFO", "LOAD", f"Loaded {len(loaded)} docs from {path}")
			except Exception as e:
				log_event("ERROR", "LOAD", f"Failed to load {path}: {e}")
	return docs

# --- Semantic Chunking ---
def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
	"""Split documents into semantic chunks for embedding."""
	splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	chunks = []
	for doc in docs:
		try:
			doc_chunks = splitter.split_documents([doc])
			chunks.extend(doc_chunks)
			log_event("INFO", "CHUNK", f"Chunked doc: {getattr(doc, 'metadata', {}).get('source', 'unknown')} into {len(doc_chunks)} chunks")
		except Exception as e:
			log_event("ERROR", "CHUNK", f"Chunking failed: {e}")
	return chunks

# --- Embedding Model Setup ---
# NOTE: Langchain's HuggingFaceEmbeddings does NOT support GGUF format. You must implement your own embedding class using llama-cpp-python or another GGUF-compatible library.
EMBED_MODEL_PATH = EMBED_MODEL_PATH

# --- GGUF Embedding Implementation ---
from llama_cpp import Llama

class GGUFEmbedder:
	"""
	Embedding wrapper for GGUF models using llama-cpp-python.
	Used for both document and query embedding.
	"""
	def __init__(self, model_path=EMBED_MODEL_PATH, embed_n_ctx=EMBED_N_CTX):
		self.model = Llama(model_path=model_path, embedding=True, n_ctx=embed_n_ctx)

	def embed_documents(self, texts):
		"""Return a list of embeddings, one per text."""
		return [self.model.embed(text) for text in texts]

	def embed_query(self, text):
		"""Return a single embedding for the query."""
		return self.model.embed(text)

def get_embedder():
	"""Return a GGUFEmbedder instance with config parameters."""
	return GGUFEmbedder(model_path=EMBED_MODEL_PATH, embed_n_ctx=EMBED_N_CTX)
# --- Qdrant DB Setup ---
def setup_qdrant():
	"""Initialize Qdrant DB and ensure collection exists with correct vector size."""
	client = qdrant_client.QdrantClient(path=DB_PATH)
	log_event("INFO", "QDRANT", f"Qdrant DB initialized at {DB_PATH}")
	try:
		if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
			client.create_collection(
				collection_name=COLLECTION_NAME,
				vectors_config={"size": VECTOR_SIZE, "distance": "Cosine"}
			)
			log_event("INFO", "QDRANT", f"Created collection '{COLLECTION_NAME}' with size {VECTOR_SIZE} and Cosine distance.")
		else:
			log_event("INFO", "QDRANT", f"Collection '{COLLECTION_NAME}' already exists.")
	except Exception as e:
		log_event("ERROR", "QDRANT", f"Failed to create/check collection: {e}")
	return client

# --- Vectorstore Write ---
def write_to_qdrant(chunks, embedder, client):
	"""
	Embed all chunks and upsert them to Qdrant DB as points.
	Each point contains the embedding vector and metadata.
	"""
	try:
		texts = [chunk.page_content for chunk in chunks]
		embeddings = embedder.embed_documents(texts)
		points = []
		for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
			points.append(PointStruct(
				id=str(uuid.uuid4()),
				vector=embedding,
				payload={
					"page_content": chunk.page_content,
					"source": chunk.metadata.get("source", "unknown"),
					"chunk_id": i
				}
			))
		client.upsert(
			collection_name=COLLECTION_NAME,
			wait=True,
			points=points
		)
		log_event("INFO", "QDRANT", f"Upserted {len(points)} points to Qdrant DB")
		return True
	except Exception as e:
		log_event("ERROR", "QDRANT", f"Failed to write to Qdrant: {e}")
		return None



# --- Main Pipeline ---
def main():
	"""
	Main pipeline entrypoint: load, chunk, embed, and upsert documents to Qdrant DB.
	"""
	log_event("INFO", "START", "Pipeline started")
	docs = load_documents()
	chunks = chunk_documents(docs)
	embedder = get_embedder()
	client = setup_qdrant()
	upsert_success = write_to_qdrant(chunks, embedder, client)
	if not upsert_success:
		log_event("ERROR", "PIPELINE", "Qdrant upsert failed. Exiting.")
		print("Error: Could not upsert to Qdrant. Check logs for details.")
		return
	log_event("INFO", "READY", "Embeddings upserted to Qdrant. Pipeline complete.")
	print("Embeddings upserted to Qdrant. Pipeline complete.")

if __name__ == "__main__":
    main()
