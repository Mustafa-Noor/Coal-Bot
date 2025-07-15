import os
import streamlit as st
from streamlit_chat import message
from huggingface_hub import InferenceClient
from modules import vector_store
from dotenv import load_dotenv

load_dotenv()

# Initialize Hugging Face Inference Client
client = InferenceClient(
    provider="novita",
    api_key=os.environ["HF_TOKEN"],
)

# Load Chroma retriever (cached)
@st.cache_resource
def get_retriever():
    db = vector_store.load_chroma_vector_store()
    return db.as_retriever(search_kwargs={"k": 3})

retriever = get_retriever()

# ---- SESSION STATE ----
st.session_state.setdefault('past', [])
st.session_state.setdefault('generated', [])

# ---- LOGIC ----
def on_input_change():
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return

    st.session_state.past.append(user_input)

    # Retrieve documents
    results = retriever.get_relevant_documents(user_input)
    context = ""
    for i, doc in enumerate(results):
        context += f"\nDocument {i+1}:\n{doc.page_content[:300]}"

    # Construct prompt
    prompt = f"""You are an assistant. Use the following documents to answer the question.

Context:
{context}

Question: {user_input}

Answer:"""

    # Generate response
    try:
        completion = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"Error: {str(e)}"

    st.session_state.generated.append({'type': 'normal', 'data': answer})


def on_btn_click():
    st.session_state.past.clear()
    st.session_state.generated.clear()

# ---- UI ----
st.title("COAL Book Chat Assistant")

chat_placeholder = st.empty()
with chat_placeholder.container():
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state['generated'][i]['data'],
            key=f"{i}_bot",
            allow_html=True,
            is_table=(st.session_state['generated'][i].get('type') == 'table')
        )

    st.button("Clear Chat", on_click=on_btn_click)

with st.container():
    st.text_input("Ask a question:", key="user_input", on_change=on_input_change)
