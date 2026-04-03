import shutil
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb


def ingest_osha_docs(docs_dir: str = "data/osha_docs", db_path: str = "chroma_db"):
    """Load OSHA text files, chunk, embed, store in ChromaDB."""
    if Path(db_path).exists():
        shutil.rmtree(db_path)

    docs_path = Path(docs_dir)
    all_texts = []
    all_metadatas = []

    for txt_file in docs_path.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        if text.strip():
            all_texts.append(text)
            all_metadatas.append({"source": txt_file.name, "standard": txt_file.stem})
            print(f"Loaded {txt_file.name} ({len(text)} chars)")

    print(f"Loaded {len(all_texts)} documents")

    # chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    chunk_metas = []
    for text, meta in zip(all_texts, all_metadatas):
        splits = splitter.split_text(text)
        for s in splits:
            if s.strip():
                chunks.append(s)
                chunk_metas.append(meta)
    print(f"Split into {len(chunks)} chunks")

    # embed
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = embed_model.embed_documents(chunks)
    print(f"Generated {len(embeddings)} embeddings")

    # store
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("osha_standards", metadata={"hnsw:space": "cosine"})

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=[f"chunk_{j}" for j in range(i, end)],
            documents=chunks[i:end],
            embeddings=embeddings[i:end],
            metadatas=chunk_metas[i:end],
        )

    print(f"Stored {collection.count()} chunks in ChromaDB at {db_path}")


if __name__ == "__main__":
    ingest_osha_docs()
