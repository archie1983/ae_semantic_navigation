import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime

class VectorDB:
    def __init__(self):
        # 1. Set up a persistent client
        self.client = chromadb.PersistentClient(path="../chroma_db")  # Data saved here

        # 2. Create or get a collection
        self.collection = self.client.get_or_create_collection(
            name="image_embeddings",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
        )

    def add_embeding(self, embedding_vector, detected_items):
        unique_id = "img_" + str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        self.collection.add(
            ids=[unique_id],  # Must be unique
            embeddings=[embedding_vector],
            metadatas=[{"detected_items": list(detected_items), "timestamp": timestamp}],
            documents=["Optional: any text description"]  # 'documents' is also optional for storing related text
        )

    def store_door_transition(self, embedding_vector, room_from, room_to):
        unique_id = "dt_" + str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        self.collection.add(
            ids=[unique_id],  # Must be unique
            embeddings=[embedding_vector],
            metadatas=[{"room_from": room_from, room_to: room_to}],
            #documents=["Optional: any text description"]  # 'documents' is also optional for storing related text
        )

    def del_embedding(self, id_to_del):
        # Delete a record by its ID
        self.collection.delete(ids=[id_to_del])
        print("Embedding deleted successfully!")

    def qry_by_item(self, item):
        # Query with your current view's embedding
        query_vector = [0.15, 0.25, 0.35]

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=5,  # Return top 5 most similar
            # Optional: filter by metadata
            # where={"detected_item": "OPENDOOR"}
            where={"detected_items": {"$contains": item}}  # Chroma's contains operator
        )

        # The results will be a dictionary containing 'ids', 'distances', and 'metadatas' lists
        for id, distance, metadata in zip(results['ids'][0], results['distances'][0], results['metadatas'][0]):
            similarity = 1 - distance  # For cosine distance, this gives you cosine similarity
            print(f"Found similar image ID: {id}, Similarity: {similarity:.4f}, Metadata: {metadata}")
            print(metadata['detected_items'])

    def del_afew_records(self):
        # Retrieve the first 5 records from the collection
        results = self.collection.get(limit=5)

        # This will return a flat list of records without similarity distances
        for idx, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][idx] if results['metadatas'] else None
            print(f"ID: {doc_id}, Metadata: {metadata}")
            self.del_embedding(doc_id)


if __name__ == "__main__":
    vdb = VectorDB()
    vdb.del_embedding("image_001")
    vdb.del_embedding("image_002")
    vdb.add_embeding([0.1, 0.2, 0.3], detected_items={"CHAIR", "TABLE", "PLATE"})
    vdb.qry_by_item("TABLE")
    vdb.del_afew_records()