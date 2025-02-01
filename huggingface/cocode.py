import streamlit as st
import os
import tempfile
import re
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")

def parse_mcqs(text):
    """Parse the LLM output into structured MCQ format"""
    questions = []
    current_question = {}
    
    # Split into individual questions
    raw_questions = text.split('\n\n')
    
    for raw_q in raw_questions:
        lines = raw_q.strip().split('\n')
        if not lines[0].startswith('Q'):
            continue
            
        question = {
            'question': lines[0].split('. ')[1].strip(),
            'options': {},
            'correctAnswer': ''
        }
        
        # Parse options
        for line in lines[1:]:
            if line.startswith(('A)', 'B)', 'C)', 'D)')):
                option = line[0]
                content = line[2:].strip()
                question['options'][option] = content
            elif line.startswith('Correct Answer:'):
                question['correctAnswer'] = line.split(': ')[1].strip()
        
        questions.append(question)
    
    return questions

# Set page configuration
st.set_page_config(page_title="Interactive MCQ Generator", layout="wide")

# Custom CSS
st.markdown("""
<style>
.stButton button {
    width: 100%;
    text-align: left;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
.correct {
    background-color: #d1fae5 !important;
    border-color: #059669 !important;
}
.incorrect {
    background-color: #fee2e2 !important;
    border-color: #dc2626 !important;
}
.feedback {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
}
.feedback.correct {
    background-color: #d1fae5;
    color: #065f46;
}
.feedback.incorrect {
    background-color: #fee2e2;
    color: #991b1b;
}
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("📚 Interactive MCQ Generator")
st.markdown("---")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define MCQ generation prompt
prompt = ChatPromptTemplate.from_template("""
Generate 5 engaging multiple-choice questions (MCQs) from the provided text. The questions should:
- Be clear and contextually relevant
- Have four answer choices (A, B, C, D)
- Include only one correct answer
- Be engaging and easy to understand

Input Text: {context}

Format each question EXACTLY as follows (maintain this format strictly):
Q1. [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct Answer: [A/B/C/D]

[Repeat for all 5 questions]
""")

# File upload
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    try:
        with st.spinner("Processing document and generating questions..."):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            # Load and process the document
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            os.unlink(temp_path)
            
            # Split text and create embeddings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs[:50])
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Create chains and generate MCQs
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vectorstore.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": "Generate MCQs from this text"})
            
            # Parse MCQs
            questions = parse_mcqs(response["answer"])
            
            # Store questions in session state
            st.session_state.questions = questions
            st.session_state.current_question = 0
            st.session_state.score = 0
            st.session_state.attempted = set()

        # Display MCQs
        if 'questions' in st.session_state:
            st.markdown("### 📝 Quiz Time!")
            
            # Create columns for the quiz interface
            col1, col2 = st.columns([7, 3])
            
            with col1:
                current_q = st.session_state.questions[st.session_state.current_question]
                st.markdown(f"**Question {st.session_state.current_question + 1} of {len(st.session_state.questions)}**")
                st.markdown(f"### {current_q['question']}")
                
                for option in ['A', 'B', 'C', 'D']:
                    question_key = f"q_{st.session_state.current_question}_{option}"
                    if st.button(
                        f"{option}) {current_q['options'][option]}", 
                        key=question_key,
                        disabled=st.session_state.current_question in st.session_state.attempted
                    ):
                        st.session_state.attempted.add(st.session_state.current_question)
                        if option == current_q['correctAnswer']:
                            st.markdown(
                                '<div class="feedback correct">✅ Correct!</div>', 
                                unsafe_allow_html=True
                            )
                            st.session_state.score += 1
                        else:
                            st.markdown(
                                f'<div class="feedback incorrect">❌ Incorrect. The correct answer was {current_q["correctAnswer"]}</div>', 
                                unsafe_allow_html=True
                            )
            
            with col2:
                st.markdown("### Score")
                st.markdown(f"**{st.session_state.score}/{len(st.session_state.questions)}**")
                
                # Navigation buttons
                col3, col4 = st.columns(2)
                with col3:
                    if st.button("Previous", 
                               disabled=st.session_state.current_question == 0):
                        st.session_state.current_question -= 1
                        st.experimental_rerun()
                
                with col4:
                    if st.button("Next", 
                               disabled=st.session_state.current_question == len(st.session_state.questions) - 1):
                        st.session_state.current_question += 1
                        st.experimental_rerun()
                
                # Show source text in expander
                with st.expander("View Source Text"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"Excerpt {i+1}:")
                        st.write(doc.page_content)
                        st.write("---")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload a PDF document to generate interactive MCQs.")