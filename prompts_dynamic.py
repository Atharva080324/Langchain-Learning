from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

st.header("Research Tool")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Inputs
paper_input = st.selectbox(
    "Select Research Paper Name", 
    [
        "Attention Is All You Need", 
        "BERT: Pre-training of Deep Bidirectional Transformers", 
        "GPT-3: Language Models are Few-Shot Learners", 
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style", 
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length", 
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Prompt Template
template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="""Please summarize the research paper titled {paper_input} with the following specifications:

Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  

2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  

Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""
)

# Generate summary when button is clicked
if st.button("Summarize"):
    formatted_prompt = template.format(
        paper_input=paper_input,
        style_input=style_input,
        length_input=length_input
    )
    result = model.invoke(formatted_prompt)
    st.subheader("Summary")
    st.write(result.content)

#Why use PromptTemplate over fstrings?
#1.Default Validation
#2.Reusable(save prompts in json format)
#3.Langchain Ecosystem
