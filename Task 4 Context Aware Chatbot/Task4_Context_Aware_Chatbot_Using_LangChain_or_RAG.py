# from langchain_community.document_loaders import DirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import LlamaCpp
# from langchain.chains import RetrievalQA
#
# loader=DirectoryLoader("dataset for chatbot",glob="*.txt")
# documents=loader.load()
# print(f"Total documents loaded: {len(documents)}")
# text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
# chunks=text_splitter.split_documents(documents)
# print(f"Total text chunks: {len(chunks)}")
#
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#
# vectorstore = FAISS.from_documents(chunks, embedding_model)
# vectorstore.save_local("faiss_index")
# retriever = vectorstore.as_retriever()
# print("Vector store created")
#
# llm=LlamaCpp(
#     model_path="model llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
#     n_ctx=2048,
#     n_threads=4,
#     temperature=0.5,
#     max_tokens=256,
#     verbose=True
# )
# qa_chain=RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True
# )
# print("RetrievalQA chain created successfully!")
# #EXAMPLE
# query = "What is techniques of AI ?"
# result = qa_chain.invoke({"query": query})
# print("\n📢 Answer:")
# print(result["result"])
#if you want to see how every thing work use above code else below for streamlit app
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain

def load_chat_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    try:
        vectorstore = FAISS.load_local("faiss_index", embedding_model)
    except:
        loader = DirectoryLoader("dataset for chatbot", glob="*.txt")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local("faiss_index")

    retriever = vectorstore.as_retriever()

    llm = LlamaCpp(
        model_path="model llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=4,
        temperature=0.5,
        max_tokens=256,
        verbose=False
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )
    return qa_chain
