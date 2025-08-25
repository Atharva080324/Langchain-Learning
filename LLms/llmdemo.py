from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI  

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key,)
#temperature is the creativity level of the model's responses
#model.temperature = 0.7 range is from 0.0 to 1.5+ depending of work
llm.temperature = 1.5

#max_output_tokens are the words limit for the response for Google Gemini
#max_completion_tokens for OPENAI
llm.max_output_tokens = 50

result = llm.invoke("write poem on cricket")

print(result.content)   # Output: "New Delhi"
