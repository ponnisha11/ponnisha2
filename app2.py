
pip install sentence-transformers

pip install numpy faiss-cpu

documents = []
sources = []

files = ["NLP.txt", "ai1.txt", "ml.txt"]

for file in files:
  with open(file, "r", encoding="utf8") as f:
        text = f.read()
        documents.append(text)
        sources.append(file)

pip install pymongo

from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
chunk_sources = []

for i, doc in enumerate(documents):
    split_chunks = doc.split(".")

    for chunk in split_chunks:
        if chunk.strip():
            chunks.append(chunk.strip())
            chunk_sources.append(sources[i])

from pymongo import MongoClient
client=MongoClient("mongodb+srv://ponnishanisha04_db_user:OdiH99IT4fL0yzGX@cluster0.okjgnos.mongodb.net/")
db=client["ex3"]
collection=db["rag3"]
document={
    "chunk":chunk_sources,
    "embedding": embed.tolist()
}
print("documents can be stored")

import faiss
query=input(" ques:")
query_embed=model.encode([query])
dim=embed.shape[1]
index=faiss.IndexFlatL2(dim)
index.add(embed)
k=1
distances,indices=index.search(query_embed1, k)
print(chunks[indices[0][0]]) 