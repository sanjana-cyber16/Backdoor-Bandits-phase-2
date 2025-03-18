import streamlit as st
import pandas as pd
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import time
import numpy as np
import re
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Initialize Google AI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your Google API Key in the .env file")
    st.stop()

# Banking knowledge base
BANKING_KNOWLEDGE = """
A bank is a financial institution licensed to receive deposits and make loans. Banks may also provide financial services such as wealth management, currency exchange, and safe deposit boxes.

Common Banking Services:
1. Savings and Checking Accounts
2. Debit and Credit Cards
3. Personal Loans
4. Mortgages
5. Investment Services
6. Online Banking

Popular Indian Banks:
- SBI (State Bank of India): Largest public sector bank
- HDFC Bank: Leading private sector bank
- ICICI Bank: Major private sector bank
- PNB (Punjab National Bank): Major public sector bank
- BOB (Bank of Baroda): Large public sector bank

Common Banking Terms:
- IFSC Code: Indian Financial System Code
- KYC: Know Your Customer
- UPI: Unified Payments Interface
- NEFT: National Electronic Funds Transfer
- RTGS: Real Time Gross Settlement
"""

# PII detection patterns
PII_PATTERNS = {
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phone': r'\b\d{10}\b|\+\d{1,3}\s?\d{10}\b',
    'pan': r'[A-Z]{5}[0-9]{4}[A-Z]',
    'account_number': r'\b\d{9,18}\b',
    'aadhar': r'\b\d{4}\s?\d{4}\s?\d{4}\b'
}

def contains_pii(text):
    """Check if text contains any PII"""
    for pattern_name, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            return True
    return False

def sanitize_text(text):
    """Remove or mask PII from text"""
    for pattern_name, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[MASKED {pattern_name.upper()}]', text)
    return text

# Retry decorator for API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def api_call_with_retry(func):
    try:
        return func()
    except Exception as e:
        if "429" in str(e):
            st.warning("API rate limit reached. Retrying after a short delay...")
            time.sleep(2)
            raise e
        raise e

# Load and prepare the dataset
@st.cache_data
def load_data():
    progress_text = "Loading dataset..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # Read the CSV file
        df = pd.read_csv('custom_data/final_dataset_trimmed.csv')
        my_bar.progress(25, text="Processing data...")
        
        # Take a smaller sample and ensure diverse bank representation
        sample_size = 500  # Reduced sample size for faster processing
        banks = df['Bank_Name'].unique()
        
        sampled_data = []
        for bank in banks:
            bank_data = df[df['Bank_Name'] == bank]
            bank_sample = bank_data.sample(n=min(len(bank_data), sample_size // len(banks)))
            sampled_data.append(bank_sample)
        
        df_sample = pd.concat(sampled_data)
        my_bar.progress(50, text="Preparing text data...")
        
        # Convert to text format more efficiently
        text_data = []
        
        # Process in smaller batches
        batch_size = 50
        total_batches = len(df_sample) // batch_size + 1
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df_sample))
            batch = df_sample.iloc[start_idx:end_idx]
            
            batch_texts = []
            for _, row in batch.iterrows():
                text = f"""
                Bank: {row['Bank_Name']}
                Transaction: Amount={row['Transaction_Amount']}, Method={row['Payment_Method']}, Frequency={row['Transaction_Frequency']}
                Location: {row['Location']}, Status: {row['Account_Status']}
                IFSC: {row['IFSC_Code']}
                """
                batch_texts.append(text)
            
            text_data.extend(batch_texts)
            progress = min(50 + int(45 * (batch_idx + 1) / total_batches), 95)
            my_bar.progress(progress, text=f"Processing batch {batch_idx + 1}/{total_batches}")
        
        # Add banking knowledge
        text_data.append(BANKING_KNOWLEDGE)
        my_bar.progress(100, text="Dataset prepared successfully!")
        time.sleep(0.5)
        my_bar.empty()
        
        return text_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Create vector store with batching
@st.cache_resource
def create_vector_store(text_chunks):
    progress_text = "Creating vector store..."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # Initialize embeddings
        my_bar.progress(10, text="Initializing embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Process in smaller batches
        batch_size = 20
        total_batches = len(text_chunks) // batch_size + 1
        
        # Initialize with first batch
        first_batch = text_chunks[:batch_size]
        my_bar.progress(20, text="Creating initial vector store...")
        vector_store = FAISS.from_texts(first_batch, embeddings)
        
        # Add remaining batches
        for i in range(1, total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(text_chunks))
            batch = text_chunks[start_idx:end_idx]
            
            if batch:
                vector_store.add_texts(batch)
            
            progress = min(20 + int(75 * (i + 1) / total_batches), 95)
            my_bar.progress(progress, text=f"Processing batch {i + 1}/{total_batches}")
        
        my_bar.progress(100, text="Vector store created successfully!")
        time.sleep(0.5)
        my_bar.empty()
        
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Custom prompt template
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question that captures all necessary context from the chat history.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:""")

QA_PROMPT = PromptTemplate.from_template("""
You are an expert banking assistant with deep knowledge of banking services, products, and operations. Use the following pieces of context to answer the question at the end.

Important Security Rules:
1. Never reveal any personal or sensitive information about customers
2. Do not provide specific account details or transaction information
3. If asked about PII (Personal Identifiable Information), politely decline
4. Focus on general banking information and procedures

Context: {context}

Question: {question}

Helpful Answer: Let me help you with that banking query.""")

# Create conversation chain
def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        max_retries=3,
        retry_wait_seconds=2
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        chain_type="stuff",
        verbose=True
    )
    return conversation_chain

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    with st.spinner('Initializing the banking assistant...'):
        # Load and process data
        text_chunks = load_data()
        if text_chunks:
            st.session_state.vector_store = create_vector_store(text_chunks)
            if st.session_state.vector_store:
                st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
                st.success('Banking Assistant is ready to help you!')
            else:
                st.error('Failed to initialize the vector store. Please try refreshing the page.')
        else:
            st.error('Failed to load the dataset. Please check the data file and try again.')

# UI Elements
st.title("🏦 Banking Assistant")
st.write("I'm your banking expert! Ask me about:")
st.write("- Bank accounts and services")
st.write("- Transaction details and patterns")
st.write("- Banking terms and procedures")
st.write("- Indian banks and their services")

# User input
user_question = st.chat_input("Ask your banking question...")

# Handle user input
if user_question and st.session_state.conversation:
    # Check for PII in the question
    if contains_pii(user_question):
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(
            "I apologize, but I cannot process questions containing personal or sensitive information. "
            "Please ask general banking questions without including any personal details, account numbers, "
            "or other sensitive information."
        )
    else:
        st.chat_message("user").write(user_question)
        
        try:
            with st.spinner('Finding the best answer for you...'):
                def get_response():
                    return st.session_state.conversation.invoke({
                        "question": user_question
                    })
                
                response = api_call_with_retry(get_response)
            
            # Sanitize the response before displaying
            sanitized_answer = sanitize_text(response["answer"])
            st.session_state.chat_history.append((user_question, sanitized_answer))
            
            with st.chat_message("assistant"):
                st.write(sanitized_answer)
        except Exception as e:
            if "429" in str(e):
                st.error("The service is currently experiencing high demand. Please try again in a few moments.")
            else:
                st.error(f"Error processing your question: {str(e)}")
            st.write("Please try asking your question again.")

# Display chat history
for message in st.session_state.chat_history:
    user_msg, ai_msg = message
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(ai_msg) 