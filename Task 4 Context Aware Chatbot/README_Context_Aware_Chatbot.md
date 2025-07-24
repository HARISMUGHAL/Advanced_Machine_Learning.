
# 🧠 Context-Aware AI Chatbot (LangChain + LlamaCpp + Streamlit)

This project is a **private document-aware chatbot** that can answer questions based on your local `.txt` files. It uses:

- 🔗 LangChain for retrieval-based QA
- 🧬 HuggingFace `all-MiniLM-L6-v2` embeddings
- 🦙 LlamaCpp (TinyLLaMA) for local LLM inference
- 🎛️ FAISS for vector storage
- 💻 Streamlit for interactive UI

---
## 🚀 How It Works

1. Loads local `.txt` files using `DirectoryLoader`
2. Splits text into manageable chunks with `RecursiveCharacterTextSplitter`
3. Converts chunks to embeddings using HuggingFace (`all-MiniLM-L6-v2`)
4. Stores embeddings in a FAISS vector index
5. Loads `TinyLLaMA` using `LlamaCpp` and builds a `ConversationalRetrievalChain`
6. Uses full chat history as context in each new query
7. Streamlit UI allows interactive question-answering

---

## 🧠 Features

- ✅ Context-aware conversation (remembers past queries)
- ✅ Works **offline** using local LLM (TinyLLaMA)
- ✅ Only uses your own `.txt` files
- ✅ Interactive UI with chat history display
- ✅ Auto builds vector store if not found

---
## 🧪 Screenshot

![Chatbot Screenshot](a4b199fb-efcf-4107-8e17-1954eb37c225.png)

---

## 📁 Folder Structure

```
.
├── dataset for chatbot/          
├── model llm/                   
│   └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
├── faiss_index/                 
├── chatbot_backend.py          
├── streamlit_app.py            
├── Task4_Context_Aware_Chatbot_Using_LangChain_or_RAG.py  
└── README.md
```

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
Recommended packages:

langchain
langchain-community
langchain-huggingface
streamlit
faiss-cpu
transformers
llama-cpp-python

Optional (for better file detection):

bash
pip install python-magic-bin


```bash
streamlit run streamlit_app.py
```

---
📌 Example Usage
Ask:
What are the techniques of AI?
Chatbot (based on local text files):

AI techniques include machine learning, deep learning, expert systems, and evolutionary algorithms...
🔐 Notes
All processing is done locally — no data leaves your machine.

Works best with well-structured .txt files in dataset for chatbot/

LLaMA model should be in model llm/ as a .gguf file.

## 📌 Features

- ✅ Offline Q&A from your local documents
- ✅ Memory of full chat context
- ✅ Fast, local inference using GGUF + LlamaCpp
- ✅ Clean Streamlit interface

---

## 👨‍💻 Author

Built with ❤️ by Haris Mughal
