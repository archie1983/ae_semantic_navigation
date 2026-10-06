import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime

class VectorDB:
    def __init__(self):
        # 1. Set up a persistent client
        self.client = chromadb.PersistentClient(path="../chroma_db")  # Data saved here
        self.create_collections()

    def create_collections(self):
        # 2. Create or get a collection
        self.room_transition_collection = self.client.get_or_create_collection(
            name="room_transitions",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
        )

        # for single door image embeddings - to identify doors before we recognise a known path
        self.door_id_collection = self.client.get_or_create_collection(
            name="door_pics",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for search
        )

    def reset_database(self):
        """Call this manually to truncate and clear out all debug data cleanly."""
        try:
            self.client.delete_collection(name="room_transitions")
            self.client.delete_collection(name="door_pics")
            print("Old collections dropped.")
        except Exception:
            print("Collection didn't exist.")

        self.create_collections()
        print("Fresh room_transitions and door_pics collections initialized.")

    def add_embeding(self, embedding_vector, detected_items):
        unique_id = "img_" + str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        self.room_transition_collection.add(
            ids=[unique_id],  # Must be unique
            embeddings=[embedding_vector],
            metadatas=[{"detected_items": list(detected_items), "timestamp": timestamp}],
            documents=["Optional: any text description"]  # 'documents' is also optional for storing related text
        )

    def store_door_transition(self, embedding_vector, room_from, room_to, early_or_late):
        unique_id = "dt_" + str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        self.room_transition_collection.add(
            ids=[unique_id],  # Must be unique
            embeddings=[embedding_vector.tolist()],
            metadatas=[{"room_from": room_from, "room_to": room_to, "early_or_late": early_or_late}],
            #documents=["Optional: any text description"]  # 'documents' is also optional for storing related text
        )

    def qry_door_transition(self, embedding_vector):
        # Safeguard: If the collection is empty, querying it will throw an error or return empty lists
        if self.room_transition_collection.count() == 0:
            print("Database is empty. No transitions to query.")
            return []

        results = self.room_transition_collection.query(
            query_embeddings=[embedding_vector.tolist()],
            n_results=min(5, self.room_transition_collection.count()) # Avoid asking for more items than exist
        )

        qry_results = []

        # Safe unpack check in case Chroma returns empty results structures
        if not results['ids'] or not results['ids'][0]:
            return qry_results

        # Chroma returns a list of lists. Index [0] gets the results for our single query vector.
        # This preserves Chroma's automatic highest-similarity-first sorting!
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        # The results will be a dictionary containing 'ids', 'distances', and 'metadatas' lists
        for id, distance, metadata in zip(ids, distances, metadatas):
            similarity = 1 - distance  # For cosine distance, this gives you cosine similarity
            print(f"Found similar TRANSITION path ID: {id}, Similarity: {similarity:.4f}, Metadata: {metadata}")
            print(metadata.get('room_from'), " TO ", metadata.get('room_to'), " early_or_late: ", metadata.get('early_or_late'))
            qry_results.append({
                'room_from': metadata.get('room_from'),
                'room_to': metadata.get('room_to'),
                'early_or_late': metadata.get('early_or_late'),
                'similarity': similarity
            })

        return qry_results

    def store_doors_imgs(self, door_imgs_embeddings, room_from, room_to, door_bboxes):
        """
        Store single door embeddings to recognize doors in the future
        :param door_imgs_embeddings:
        :param room_from:
        :param room_to:
        :param door_bboxes:
        :return:
        """
        unique_id = "di_" + str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        door_infos = zip(door_imgs_embeddings, door_bboxes)

        record_cnt = 0
        for di in door_infos:
            door_pic_embedding = di[0]
            bbox = di[1]
            unique_id_to_use = unique_id + "_" + str(record_cnt)

            self.door_id_collection.add(
                ids=[unique_id_to_use],  # Must be unique
                embeddings=[door_pic_embedding.tolist()],
                metadatas=[{"room_from": room_from, "room_to": room_to, "bbox": bbox}]
            )
            record_cnt += 1

    def qry_door_img(self, embedding_vector):
        """
        Query single door embeddings for ID before actual path transition.
        :param embedding_vector:
        :return:
        """
        # Safeguard: If the collection is empty, querying it will throw an error or return empty lists
        if self.door_id_collection.count() == 0:
            print("Database is empty. No transitions to query.")
            return []

        results = self.door_id_collection.query(
            query_embeddings=[embedding_vector.tolist()],
            n_results=min(5, self.door_id_collection.count()) # Avoid asking for more items than exist
        )

        qry_results = []

        # Safe unpack check in case Chroma returns empty results structures
        if not results['ids'] or not results['ids'][0]:
            return qry_results

        # Chroma returns a list of lists. Index [0] gets the results for our single query vector.
        # This preserves Chroma's automatic highest-similarity-first sorting!
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        # The results will be a dictionary containing 'ids', 'distances', and 'metadatas' lists
        for id, distance, metadata in zip(ids, distances, metadatas):
            similarity = 1 - distance  # For cosine distance, this gives you cosine similarity
            print(f"Found similar DOOR image ID: {id}, Similarity: {similarity:.4f}, Metadata: {metadata}")
            print(metadata.get('room_from'), " TO ", metadata.get('room_to'), " bbox: ", metadata.get('bbox'))
            qry_results.append({
                'room_from': metadata.get('room_from'),
                'room_to': metadata.get('room_to'),
                'bbox': metadata.get('bbox'),
                'similarity': similarity
            })

        return qry_results

    def del_embedding(self, id_to_del):
        # Delete a record by its ID
        self.room_transition_collection.delete(ids=[id_to_del])
        print("Embedding deleted successfully!")

    def qry_by_item(self, item):
        # Query with your current view's embedding
        query_vector = [0.15, 0.25, 0.35]

        results = self.room_transition_collection.query(
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
        results = self.room_transition_collection.get(limit=5)

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