import streamlit as st
from Task4_Context_Aware_Chatbot_Using_LangChain_or_RAG import load_chat_chain

st.set_page_config(page_title="Context-Aware Chatbot", page_icon="🧠")
st.title("🧠 Context-Aware AI Chatbot")
st.markdown("Ask anything from your local documents. Your chat is context-aware!")

if "chain" not in st.session_state:
    with st.spinner("Loading model and vector index..."):
        st.session_state.chain = load_chat_chain()
        st.session_state.chat_history = []

user_query = st.text_input("Ask your question here:", placeholder="e.g. What are techniques of AI?", key="user_input")

if user_query:
    with st.spinner("Thinking..."):
        result = st.session_state.chain.invoke({
            "question": user_query,
            "chat_history": st.session_state.chat_history
        })

        st.session_state.chat_history.append((user_query, result["answer"]))

if st.session_state.get("chat_history"):
    st.markdown("###  Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"** You:** {q}")
        st.markdown(f"** Bot:** {a}")
        st.divider()
