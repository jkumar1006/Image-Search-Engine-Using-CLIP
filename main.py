import chromadb
from clip_embeddings import get_embedding_on_images, compute_text_embeddings, compute_image_embeddings

# Initialize ChromaDB client

def add_embeddings_to_colleection():
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)

    # Get or create a collection named 'images' in ChromaDB
    collection = chroma_client.get_or_create_collection(name='images2', metadata={"hnsw:space": "cosine"})

    # Get image embeddings and add them to the collection
    image_embeddings = get_embedding_on_images('images')
    
    collection.add(
        embeddings=[e["embedding"][0].tolist() for e in image_embeddings],
        metadatas=[{k: e[k] for k in ["filePath"]} for e in image_embeddings],
        
        ids=[e["id"] for e in image_embeddings],
    )

##add_embeddings_to_colleection()
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

print(chroma_client.list_collections())
collection = chroma_client.get_or_create_collection(name='images2', metadata={"hnsw:space": "cosine"})
print(collection.count())

# Function to perform vector search
def vector_search(emb):
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
   
    # Get or create a collection named 'images' in ChromaDB
    collection = chroma_client.get_or_create_collection(name='images2', metadata={"hnsw:space": "cosine"})
    emb = emb.tolist()
    results = collection.query(
        query_embeddings=emb,
        n_results=3
    )
    results = results["metadatas"][0]
    results = [result['filePath'] for result in results]
    return results

print("Inserted images to vector db")

flag = 1
while flag:
    print("1: To perform text search")
    print("2: To perform search using image")
    print("3: EXIT")
    user_input = int(input("Enter your choice: "))

    if user_input == 1:
        search_text = input("Enter the text: ")
        query_emb = compute_text_embeddings(search_text)
        results = vector_search(query_emb[0])
        print(results)

    elif user_input == 2:
        image_path = input("Enter the path of the image: ")
        query_emb = compute_image_embeddings(image_path)
        results = vector_search(query_emb[0])
        print(results)

    elif user_input == 3:
        flag = 0
        print("Exiting...")

    else:
        print("Select a valid option")
