from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv()

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents=[
    "Delhi is the capital of India",
    "Mumbai is the capital of maharashtra"
]

vector = embedding.embed_documents("Delhi is the capital of India")

print(str(vector)) 
//it will return the contextual meaning of the query_document in 32 dimensions in vector

