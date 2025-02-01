import requests
import streamlit as st

def get_essay_response(input_text) :
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json= {'input':{'topic': input_text}}
    )
    return response.json()['output']

def get_poem_response(input_text) :
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json= {'input':{'topic': input_text}}
    )
    return response.json()['output']

st.title("LANGCHAIN demo with LLAMA3.2 API")
input_text = st.text_input("Write me an pickup for")
input_text1 = st.text_input("Write me an poem on")

if input_text:
    st.write(get_essay_response(input_text))

if input_text1:
    st.write(get_poem_response(input_text1))