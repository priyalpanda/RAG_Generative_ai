import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

Settings.llm = Ollama(model="llama3.2", request_timeout=360.0)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
Settings.embed_model = embed_model

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)

query_engine = index.as_query_engine()
response = query_engine.query("What topics are covered in this class?")
print(response)
