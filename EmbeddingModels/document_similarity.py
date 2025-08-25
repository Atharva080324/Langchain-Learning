from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

os.environ["GOOGLE_API_KEY"] = "AIzaSyAKhMgGSKl6uh0kSCidbxmwK5FAZXrDJ8M"
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"   # Gemini embedding model
)

document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about virat kohli"

documents_embedding = embeddings.embed_documents(document)#5 vector

query_embedding = embeddings.embed_query(query)#1 vector
query_embedding = np.array(query_embedding).reshape(1, -1)

scores = cosine_similarity(query_embedding, documents_embedding)[0]#1*5

index,score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(document[index])
print("Similarity Score is ",score)
