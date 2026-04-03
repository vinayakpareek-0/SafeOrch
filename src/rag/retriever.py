from langchain_huggingface import HuggingFaceEmbeddings
import chromadb


class OSHARetriever:
    """Query ChromaDB for relevant OSHA regulation context."""

    def __init__(self, db_path: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("osha_standards")
        self.embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def query(self, text: str, top_k: int = 5) -> str:
        """Return relevant OSHA context as a single string."""
        embedding = self.embed_model.embed_query(text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        docs = results["documents"][0]
        sources = [m["source"] for m in results["metadatas"][0]]
        
        # format as context string
        context_parts = []
        for doc, src in zip(docs, sources):
            context_parts.append(f"[{src}]\n{doc}")
        
        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    r = OSHARetriever()
    result = r.query("worker not wearing hard hat on construction site")
    print(result)
