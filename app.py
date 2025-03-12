import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
import os

# Streamlit Page Configuration
st.set_page_config(page_title="🤖 Data Science Chatbot", layout="wide")
st.title("Suman Data Science AI Chatbot")
st.subheader("Ask your doubts here")

# Initialize Chat Memory in Session State
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize AI Model
chat_model = ChatGoogleGenerativeAI(
    google_api_key="AIzaSyCBhbuJbxjlghoZ3X1HQhS_qwuMpSE1wC0",  # Replace with a secure method
    model="gemini-1.5-pro",
    temperature=1
)

# Define Chat Prompt Template
chat_template = ChatPromptTemplate(
    messages=[
        ("system", "👨‍🏫 You are an AI Data Science Tutor. "
                   "You must answer ONLY Data Science-related questions. "
                   "If the user asks non-data science questions, politely refuse and redirect them. "
                   "Provide detailed explanations with examples and clean code snippets. "
                   "For visualization-related topics, generate appropriate images using AI."),
        MessagesPlaceholder(variable_name="chat_history"),  # Correctly retrieves past messages
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

output_parser = StrOutputParser()

# Function to retrieve chat history and user input
def get_history_and_input(user_input):
    return {
        "chat_history": st.session_state.memory.chat_memory.messages,  # Fetch previous messages
        "human_input": user_input
    }

chain = (
    RunnableLambda(lambda x: get_history_and_input(x["human_input"]))
    | chat_template
    | chat_model
    | output_parser
)

# Display Chat Messages
for message in st.session_state.memory.chat_memory.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# User Input
if user_input := st.chat_input("💬 Write your Message:"):
    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI Response
    query = {"human_input": user_input}
    response = chain.invoke(query)

    # Display AI Response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save Conversation History in Session State
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(response)
