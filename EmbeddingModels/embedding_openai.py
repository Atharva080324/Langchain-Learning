from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv()

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

vector = embedding.embed_query("Delhi is the capital of India")

print(str(vector)) 
//it will return the contextual meaning of the query in 32 dimensions in vector

