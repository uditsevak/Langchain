from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn 
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "A simple API Server"
)

llm = Ollama(model="llama3.2")

prompt1 = ChatPromptTemplate.from_template("Write a pickup line about {topic} around 2 lines ")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} around 70 words ")

add_routes(
    app,
    prompt1 | llm,
    path = "/essay"
)

add_routes(
    app,
    prompt2 | llm,
    path = "/poem"
)

if __name__ == "__main__" :
    uvicorn.run(app, host = "localhost", port = 8000)