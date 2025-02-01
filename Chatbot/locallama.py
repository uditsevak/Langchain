# Imports
#from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import streamlit as st
import os 
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {Question}")  # Ensure variable name matches what you will pass.
    ]
)

# Streamlit Framework
st.title("Chatbot")
input_text = st.text_input("Search the topic you want")

# Ollama LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Pass the correct variable name
if input_text:
    st.write(chain.invoke({"Question": input_text}))  # Use "question", not "Question".
