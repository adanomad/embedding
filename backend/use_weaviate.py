import time
import weaviate


class WeaviateEmbeddingStore:
    def __init__(self, weaviate_url):
        self.weaviate_url = weaviate_url
        self.client = weaviate.Client(weaviate_url)

    def store_embedding(self, file_name, md5_hash, file_size, model_name, embedding):
        data_object = {
            "fileName": file_name,
            "md5Hash": md5_hash,
            "fileSize": file_size,
            "modelName": model_name,
        }
        self.client.data_object.create(data_object, "FileEmbedding", vector=embedding)

    def get_nearest_neighbors(self, embedding, max_results=5):
        nearest_neighbors = (
            self.client.query.get(
                "FileEmbedding",
                ["fileName", "md5Hash", "fileSize", "modelName"],
            )
            .with_near_vector({"vector": embedding})
            .with_limit(max_results)
            .do()
        )
        return nearest_neighbors

    def delete_embedding(self, md5_hash):
        # Construct a query to find the object by md5Hash
        result = self.client.batch.delete_objects(
            class_name="FileEmbedding",
            where={
                "path": ["md5Hash"],
                "operator": "ContainsAny",
                "valueTextArray": [md5_hash],
            },
        )

    def delete_all_embeddings(self):
        self.client.schema.delete_class("FileEmbedding")

    def check_md5_exists(self, md5_hash, model_name):
        start_time = time.time()
        query = f"""
        {{
            Get {{
                FileEmbedding(where: {{operator: And, operands: [{{path: ["md5Hash"], operator: Equal, valueString: "{md5_hash}"}}, {{path: ["modelName"], operator: Equal, valueString: "{model_name}"}}]}}) {{
                    md5Hash
                }}
            }}
        }}
        """
        try:
            result = self.client.query.raw(query)
            duration = time.time() - start_time
            exists = bool(result["data"]["Get"]["FileEmbedding"])
            return exists, duration
        except weaviate.exceptions.UnexpectedStatusCodeException as e:
            print(f"info: {e}")
            return False, 0

    def check_multiple_md5_exists(self, md5_hashes, model_name):
        start_time = time.time()
        results = []
        for md5_hash in md5_hashes:
            exists, _ = self.check_md5_exists(md5_hash, model_name)
            results.append((md5_hash, exists))
        duration = time.time() - start_time
        return results, duration


# Usage Example
weaviate_url = "http://henderson:8080"
print("Connecting to Weaviate at", weaviate_url)
store = WeaviateEmbeddingStore(weaviate_url)

# Example data (replace with actual values)
file_name = "example.jpg"
md5_hash = "abc123"
file_size = 1024
model_name = "example_model"
# you can choose any embedding you want, or let your model generate one
embedding = [0.1, 0.2, 0.3]

# Store embedding (works)
store.store_embedding(file_name, md5_hash, file_size, model_name, embedding)

# Retrieve nearest neighbors
neighbors = store.get_nearest_neighbors(embedding)
print("neighbors", neighbors)

# Raw
query = """
{
  Get {
    FileEmbedding {
      fileName
      md5Hash
      fileSize
      modelName
    }
  }
}
"""

raw_all = store.client.query.raw(query)
print("raw_all", raw_all)


# Check a single MD5 hash
exists, query_time = store.check_md5_exists("abc123", "example_model")
print(f"Exists: {exists}, Query Time: {query_time} seconds")

# Check multiple MD5 hashes
md5_hashes = ["abc123", "def456", "ghi789"]
results, total_query_time = store.check_multiple_md5_exists(md5_hashes, "example_model")
for md5_hash, exists in results:
    print(f"MD5: {md5_hash}, Exists: {exists}")
print(f"Total Query Time for Multiple MD5s: {total_query_time} seconds")


# Delete
store.delete_all_embeddings()

raw_all = store.client.query.raw(query)
print(raw_all)

# Drop class
store.client.schema.delete_class("FileEmbedding")
print("Dropped FileEmbedding Class")

raw_all = store.client.query.raw(query)
print(raw_all)

print("Done")
