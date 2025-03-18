import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Google AI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your Google API Key in the .env file")
    st.stop()

llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-1.5-pro",
    temperature=0.2
)

st.title("💬 AI Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Write Message History
for msg in st.session_state.messages:
    st.chat_message(
        msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"
    ).write(msg["content"])

# Generator for Streaming Tokens
def generate_response():
    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    prompt = messages[-1]["content"]  # Get the last user message as the input
    response = llm.stream(prompt)
    full_response = ""
    for token in response:
        full_response += token
        yield token
    st.session_state["full_message"] = full_response

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        response = st.write_stream(generate_response())

    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state["full_message"]}
    )
