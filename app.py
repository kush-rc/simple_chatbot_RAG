# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from PyPDF2 import PdfReader
# import tempfile

# # Load environment
# load_dotenv()

# # Page config
# st.set_page_config(
#     page_title="RAG Chatbot with PDF",
#     page_icon="🤖",
#     layout="wide"
# )

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "documents" not in st.session_state:
#     st.session_state.documents = []
# if "vectorizer" not in st.session_state:
#     st.session_state.vectorizer = None
# if "doc_vectors" not in st.session_state:
#     st.session_state.doc_vectors = None

# # Initialize LLM
# @st.cache_resource
# def initialize_llm():
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         api_key = st.secrets.get("GROQ_API_KEY", "")
    
#     return ChatGroq(
#         groq_api_key=api_key,
#         model_name="llama-3.3-70b-versatile",
#         temperature=0.7
#     )

# llm = initialize_llm()

# # Function to extract text from PDF
# def extract_pdf_text(pdf_file):
#     """Extract text from PDF file"""
#     try:
#         pdf_reader = PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text() + "\n\n"
#         return text
#     except Exception as e:
#         st.error(f"Error reading PDF: {str(e)}")
#         return ""

# # Function to process documents
# def process_documents(texts):
#     """Process and vectorize documents"""
#     if not texts:
#         return None, None
    
#     # Split into chunks by paragraphs
#     all_chunks = []
#     for text in texts:
#         chunks = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
#         all_chunks.extend(chunks)
    
#     if not all_chunks:
#         return None, None
    
#     # Create TF-IDF vectors
#     vectorizer = TfidfVectorizer(max_features=1000)
#     doc_vectors = vectorizer.fit_transform(all_chunks)
    
#     return vectorizer, doc_vectors, all_chunks

# # Function to get relevant documents
# def get_relevant_docs(question, top_k=3):
#     """Find most relevant documents"""
#     if st.session_state.vectorizer is None:
#         return []
    
#     question_vector = st.session_state.vectorizer.transform([question])
#     similarities = cosine_similarity(question_vector, st.session_state.doc_vectors).flatten()
#     top_indices = similarities.argsort()[-top_k:][::-1]
    
#     return [st.session_state.documents[i] for i in top_indices]

# # Function to get answer
# def get_answer(question):
#     """Get answer using RAG"""
#     relevant_docs = get_relevant_docs(question)
    
#     if not relevant_docs:
#         return "Please upload some documents first!", []
    
#     context = "\n\n".join(relevant_docs)
    
#     prompt = f"""Based on the following context, answer the question. If you cannot answer based on the context, say so clearly.

# Context:
# {context}

# Question: {question}

# Answer:"""
    
#     response = llm.invoke(prompt)
#     return response.content, relevant_docs

# # ============ SIDEBAR ============
# with st.sidebar:
#     st.title("📁 Document Upload")
#     st.write("Upload PDFs or TXT files")
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Choose files",
#         type=["pdf", "txt"],
#         accept_multiple_files=True,
#         help="Upload PDF or TXT files to chat with"
#     )
    
#     if uploaded_files:
#         if st.button("📤 Process Files", type="primary"):
#             with st.spinner("Processing files..."):
#                 texts = []
                
#                 for uploaded_file in uploaded_files:
#                     file_type = uploaded_file.name.split('.')[-1].lower()
                    
#                     if file_type == 'pdf':
#                         # Extract PDF text
#                         text = extract_pdf_text(uploaded_file)
#                         if text:
#                             texts.append(text)
#                             st.success(f"✅ {uploaded_file.name}")
                    
#                     elif file_type == 'txt':
#                         # Read TXT file
#                         text = uploaded_file.read().decode('utf-8')
#                         texts.append(text)
#                         st.success(f"✅ {uploaded_file.name}")
                
#                 # Process all documents
#                 if texts:
#                     result = process_documents(texts)
#                     if result[0] is not None:
#                         st.session_state.vectorizer = result[0]
#                         st.session_state.doc_vectors = result[1]
#                         st.session_state.documents = result[2]
#                         st.success(f"🎉 Processed {len(st.session_state.documents)} text chunks!")
#                     else:
#                         st.error("No valid text found in documents")
    
#     # Show document stats
#     if st.session_state.documents:
#         st.divider()
#         st.metric("Documents Loaded", len(st.session_state.documents))
        
#         if st.button("🗑️ Clear All Documents"):
#             st.session_state.documents = []
#             st.session_state.vectorizer = None
#             st.session_state.doc_vectors = None
#             st.session_state.messages = []
#             st.rerun()
    
#     st.divider()
#     st.markdown("### 🔑 API Key")
#     if os.getenv("GROQ_API_KEY"):
#         st.success("✅ API Key loaded from .env")
#     else:
#         st.warning("⚠️ No API key found")

# # ============ MAIN CHAT INTERFACE ============
# st.title("🤖 RAG Chatbot with PDF Support")
# st.caption("Upload documents and ask questions!")

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
        
#         # Show sources for assistant messages
#         if message["role"] == "assistant" and "sources" in message:
#             with st.expander("📄 View Sources"):
#                 for i, source in enumerate(message["sources"], 1):
#                     st.text(f"{i}. {source[:200]}...")

# # Chat input
# if prompt := st.chat_input("Ask me anything about your documents..."):
#     # Check if documents are loaded
#     if not st.session_state.documents:
#         st.warning("⚠️ Please upload and process documents first!")
#     else:
#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Get bot response
#         with st.chat_message("assistant"):
#             with st.spinner("🤔 Thinking..."):
#                 answer, sources = get_answer(prompt)
#                 st.markdown(answer)
                
#                 # Show sources
#                 with st.expander("📄 View Sources"):
#                     for i, source in enumerate(sources, 1):
#                         st.text(f"{i}. {source[:200]}...")
        
#         # Save assistant message
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": answer,
#             "sources": sources
#         })

# # Instructions if no documents
# if not st.session_state.documents:
#     st.info("""
#     👈 **Get Started:**
#     1. Upload PDF or TXT files using the sidebar
#     2. Click "Process Files"
#     3. Start asking questions!
#     """)

# # Footer
# st.divider()
# st.caption("Powered by Groq API 🚀 | Built with Streamlit")






import streamlit as st
import os
from langchain_groq import ChatGroq
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Page config
st.set_page_config(
    page_title="Physics RAG Chatbot",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "doc_vectors" not in st.session_state:
    st.session_state.doc_vectors = None

# Initialize LLM
@st.cache_resource
def initialize_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# Function to process documents
def process_documents(texts):
    """Process and vectorize documents"""
    if not texts:
        return None, None, None

    all_chunks = []
    for text in texts:
        chunks = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
        all_chunks.extend(chunks)

    if not all_chunks:
        return None, None, None

    vectorizer = TfidfVectorizer(max_features=1000)
    doc_vectors = vectorizer.fit_transform(all_chunks)

    return vectorizer, doc_vectors, all_chunks

# Function to get relevant documents
def get_relevant_docs(question, top_k=3):
    """Find most relevant documents"""
    if st.session_state.vectorizer is None:
        return []

    question_vector = st.session_state.vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, st.session_state.doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [st.session_state.documents[i] for i in top_indices]

# Function to get answer
def get_answer(question, llm):
    """Get answer using RAG"""
    relevant_docs = get_relevant_docs(question)

    if not relevant_docs:
        return "Please upload some documents first!", []

    context = "\n\n".join(relevant_docs)

    prompt = f"""Based on the following context, answer the question. If you cannot answer based on the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content, relevant_docs

# ============ SIDEBAR ============
with st.sidebar:
    st.title("🔬 Physics RAG Bot")

    # API Key input
    st.markdown("### 🔑 Groq API Key")

    # Try to get API key from secrets first (for Streamlit Cloud)
    default_key = ""
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except:
        pass

    api_key_input = st.text_input(
        "Enter your Groq API Key",
        value=default_key,
        type="password",
        help="Get free API key from https://console.groq.com/keys"
    )

    if api_key_input:
        st.success("✅ API Key provided")
    else:
        st.warning("⚠️ Please enter API key")
        st.info("Get FREE API key: https://console.groq.com/keys")

    st.divider()

    st.title("📁 Document Upload")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload PDF or TXT files"
    )

    if uploaded_files:
        if st.button("📤 Process Files", type="primary"):
            with st.spinner("Processing files..."):
                texts = []

                for uploaded_file in uploaded_files:
                    file_type = uploaded_file.name.split('.')[-1].lower()

                    if file_type == 'pdf':
                        text = extract_pdf_text(uploaded_file)
                        if text:
                            texts.append(text)
                            st.success(f"✅ {uploaded_file.name}")

                    elif file_type == 'txt':
                        text = uploaded_file.read().decode('utf-8')
                        texts.append(text)
                        st.success(f"✅ {uploaded_file.name}")

                if texts:
                    result = process_documents(texts)
                    if result[0] is not None:
                        st.session_state.vectorizer = result[0]
                        st.session_state.doc_vectors = result[1]
                        st.session_state.documents = result[2]
                        st.success(f"🎉 Processed {len(st.session_state.documents)} chunks!")
                    else:
                        st.error("No valid text found")

    if st.session_state.documents:
        st.divider()
        st.metric("Documents Loaded", len(st.session_state.documents))

        if st.button("🗑️ Clear All"):
            st.session_state.documents = []
            st.session_state.vectorizer = None
            st.session_state.doc_vectors = None
            st.session_state.messages = []
            st.rerun()

# ============ MAIN CHAT ============
st.title("🔬 Physics RAG Chatbot")
st.caption("Upload your Physics PDF and ask questions!")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.text(f"{i}. {source[:200]}...")

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    if not api_key_input:
        st.error("⚠️ Please enter your Groq API key in the sidebar!")
    elif not st.session_state.documents:
        st.warning("⚠️ Please upload and process documents first!")
    else:
        # Initialize LLM with API key
        llm = initialize_llm(api_key_input)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer, sources = get_answer(prompt, llm)
                st.markdown(answer)

                with st.expander("📄 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.text(f"{i}. {source[:200]}...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

if not st.session_state.documents:
    st.info("""
    👈 **Get Started:**
    1. Enter your Groq API key (free from console.groq.com)
    2. Upload your Physics PDF
    3. Click "Process Files"
    4. Start asking questions!
    """)

st.divider()
st.caption("Powered by Groq API 🚀 | Built with Streamlit")
