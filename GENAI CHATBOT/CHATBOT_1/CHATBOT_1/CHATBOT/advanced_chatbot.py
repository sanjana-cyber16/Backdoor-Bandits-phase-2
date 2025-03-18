import streamlit as st
import os
import time
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Advanced AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border-left: 5px solid #2b6cb0;
    }
    .chat-message.bot {
        background-color: white;
        border-left: 5px solid #4299e1;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .stButton button {
        background-color: #4299e1;
        color: white;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2b6cb0;
    }
    .sidebar .stButton button {
        width: 100%;
    }
    .stTextInput input {
        border-radius: 0.25rem;
        border: 1px solid #e2e8f0;
        padding: 0.5rem;
    }
    h1, h2, h3 {
        color: #2d3748;
    }
    .stAlert {
        border-radius: 0.25rem;
    }
    footer {
        visibility: hidden;
    }
    .viewerBadge_container__r5tak {
        display: none;
    }
    .stSpinner {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Constants
VECTOR_STORE_PATH = "faiss_index_advanced"
CUSTOM_DATA_DIR = "custom_data"
PRIMARY_DATASET = "final dataset trimmed.csv"
PROTECTED_FILES = [PRIMARY_DATASET]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'kb_created' not in st.session_state:
    st.session_state.kb_created = False
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'api_error' not in st.session_state:
    st.session_state.api_error = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/chat--v1.png", width=80)
    st.title("Advanced AI Chatbot")
    
    # API Key input
    api_key = st.text_input("Google API Key", value=os.environ.get("GOOGLE_API_KEY", ""), type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    
    # Admin section
    with st.expander("Admin Settings"):
        admin_password = st.text_input("Admin Password", type="password")
        if st.button("Login as Admin"):
            if admin_password == "admin123":
                st.session_state.is_admin = True
                st.success("Admin access granted")
            else:
                st.error("Invalid password")
        
        if st.session_state.is_admin:
            st.success("Admin mode active")
            if st.button("Reset Knowledge Base"):
                with st.spinner("Resetting..."):
                    if os.path.exists(VECTOR_STORE_PATH):
                        import shutil
                        shutil.rmtree(VECTOR_STORE_PATH)
                    st.session_state.kb_created = False
                    st.success("Knowledge base reset")
    
    st.divider()
    
    # About section
    st.markdown("### About")
    st.markdown("""
    This advanced chatbot uses Google's Generative AI and vector embeddings to provide accurate answers based on your project data.
    
    The chatbot is trained on your custom dataset and can answer questions about your project.
    """)
    
    st.divider()
    
    # Footer
    st.caption("© 2024 Advanced AI Chatbot")

# Main content
st.title("Project Knowledge Assistant")

# Initialize Google Generative AI
@st.cache_resource(show_spinner=False)
def initialize_genai():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key not found. Please enter your API key in the sidebar.")
        return None, None
    
    try:
        genai.configure(api_key=api_key)
        
        # Get available models
        models = genai.list_models()
        chat_models = [m.name for m in models if "generateContent" in m.supported_generation_methods]
        
        # Find a suitable model
        if not chat_models:
            st.warning("No chat models found. Using default model.")
            model_name = "models/gemini-1.5-flash"
        else:
            model_name = next((m for m in chat_models if "gemini" in m and not "vision" in m), chat_models[0])
        
        st.session_state.model_name = model_name
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=0.2,
        )
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/embedding-001"
        )
        
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing Google Generative AI: {str(e)}")
        st.session_state.api_error = str(e)
        return None, None

# Load and process data
def load_dataset():
    documents = []
    
    # Check for primary dataset
    primary_path = os.path.join(CUSTOM_DATA_DIR, PRIMARY_DATASET)
    
    # Create directory if it doesn't exist
    if not os.path.exists(CUSTOM_DATA_DIR):
        os.makedirs(CUSTOM_DATA_DIR)
    
    # Check if primary dataset exists
    if os.path.exists(primary_path):
        try:
            # Load CSV
            df = pd.read_csv(primary_path)
            
            # Convert DataFrame to documents
            for i, row in df.iterrows():
                # Combine all columns into a single text
                content = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                documents.append(Document(
                    page_content=content,
                    metadata={"source": PRIMARY_DATASET, "row": i}
                ))
            
            # Split documents if needed
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                documents = text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"Error loading primary dataset: {str(e)}")
    
    # If no documents were loaded, use default
    if not documents:
        documents = [
            Document(
                page_content="This is an AI-powered chatbot that helps answer questions using a knowledge base.",
                metadata={"source": "default"}
            ),
            Document(
                page_content="The chatbot uses Google's Generative AI for text generation and embeddings for semantic search.",
                metadata={"source": "default"}
            ),
            Document(
                page_content="To use the chatbot, simply type your question and press Enter.",
                metadata={"source": "default"}
            )
        ]
    
    return documents

# Create or load vector store
@st.cache_resource(show_spinner=False)
def get_vector_store(_embeddings):
    # Load and process the dataset
    dataset = load_dataset()
    if not dataset:
        return None
    
    if os.path.exists(VECTOR_STORE_PATH):
        # Load existing vector store
        return FAISS.load_local(VECTOR_STORE_PATH, _embeddings, allow_dangerous_deserialization=True)
    else:
        # Create new vector store
        vector_store = FAISS.from_documents(dataset, _embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        return vector_store

# Create QA chain
@st.cache_resource(show_spinner=False)
def get_qa_chain(_vector_store, _llm):
    if not _vector_store or not _llm:
        return None
    
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions about the project.

    Context: {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return chain

# Initialize AI components
with st.spinner("Initializing AI components..."):
    llm, embeddings = initialize_genai()
    vector_store = get_vector_store(embeddings)
    qa_chain = get_qa_chain(vector_store, llm)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your project..."):
    # Check for restricted phrases
    restricted_phrases = ["delete", "remove", "drop", "erase", "destroy", "clear all", "wipe"]
    
    if any(phrase in prompt.lower() for phrase in restricted_phrases) and not st.session_state.is_admin:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add assistant message
        response = "I'm sorry, but that operation is restricted. You don't have permission to perform this action."
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Check if QA chain is available
            if qa_chain is None:
                response = "I'm having trouble connecting to the AI service. Please check your API key and try again."
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                try:
                    # Get response from QA chain
                    with st.spinner("Thinking..."):
                        result = qa_chain.invoke({"query": prompt})
                        response = result["result"]
                    
                    # Simulate typing
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.01)
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Final response
                    message_placeholder.markdown(response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"question": prompt, "answer": response})
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = str(e)
                    st.session_state.api_error = error_msg
                    response = f"I encountered an error: {error_msg}"
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

# Display model info
if st.session_state.model_name:
    st.caption(f"Using model: {st.session_state.model_name}")

# Error display
if st.session_state.api_error:
    with st.expander("Troubleshooting"):
        st.error(f"Last error: {st.session_state.api_error}")
        st.markdown("""
        Common solutions:
        1. Check your API key
        2. Ensure you have access to Google Generative AI models
        3. Check your internet connection
        """) 