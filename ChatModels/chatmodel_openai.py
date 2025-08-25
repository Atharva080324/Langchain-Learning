from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4")

#temperature is the creativity level of the model's responses
#model.temperature = 0.7 range is from 0.0 to 1.5+ depending of work

result = model.invoke("What is the capital of India?")
print(result)