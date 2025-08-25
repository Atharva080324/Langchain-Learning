from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-20240922",temperature=0.7)

result = model.invoke("write a poem about cricket")
print(result.content)

