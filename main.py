import os
import json
import html
from datetime import datetime
import streamlit as st
import markdown
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Custom exception classes for document ingestion
class DuplicateUploadError(Exception):
    pass

class EmptyPDFError(Exception):
    pass

class CorruptedPDFError(Exception):
    pass

class EmbeddingFailureError(Exception):
    pass

class ChromaDBFailureError(Exception):
    pass

working_dir = os.path.dirname(os.path.abspath(__file__))

config_data = json.load(
    open(f"{working_dir}/config.json")
)

GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def get_indexed_documents(vectorstore):
    """
    Retrieves the list of unique document basenames currently indexed in the vector store.
    """
    try:
        metadatas = vectorstore._collection.get()['metadatas']
        if not metadatas:
            return []
        sources = set()
        for m in metadatas:
            if m and 'source' in m:
                sources.add(os.path.basename(m['source']))
        return sorted(list(sources))
    except Exception as e:
        st.sidebar.error(f"Error reading vector database: {str(e)}")
        return []

def delete_document_from_vectorstore(vectorstore, doc_name):
    try:
        collection_data = vectorstore._collection.get()
        ids = collection_data.get('ids', [])
        metadatas = collection_data.get('metadatas', [])
        
        ids_to_delete = []
        for idx, meta in zip(ids, metadatas):
            if meta and 'source' in meta:
                meta_doc_name = os.path.basename(meta['source'])
                if meta_doc_name == doc_name:
                    ids_to_delete.append(idx)
                    
        if ids_to_delete:
            vectorstore._collection.delete(ids=ids_to_delete)
            if hasattr(vectorstore, "persist"):
                vectorstore.persist()
            return True
        return False
    except Exception as e:
        st.sidebar.error(f"Error deleting document from database: {str(e)}")
        return False

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=st.session_state.embeddings
    )

    return vectorstore

def chat_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    retriever = vectorstore.as_retriever()

    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

    system_template = (
        "You are a helpful document assistant. Answer questions naturally and directly. "
        "Explain concepts in simple language. Do not begin answers with phrases like "
        "'According to the text' or 'The text states'. Summarize information clearly and concisely.\n\n"
        "Context:\n{context}"
    )
    
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return chain

def save_uploaded_files(uploaded_files):
    """
    Saves uploaded files to the temporary folder 'uploaded_docs/'.
    Checks for duplicates against st.session_state.indexed_documents.
    Raises DuplicateUploadError or ValueError on empty/duplicate files.
    """
    upload_dir = os.path.join(working_dir, "uploaded_docs")
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        
        # Check if already indexed
        if filename in st.session_state.get("indexed_documents", []):
            raise DuplicateUploadError(f"Duplicate upload detected: '{filename}' is already indexed.")
            
        # Check if empty (0 bytes)
        if uploaded_file.size == 0:
            raise EmptyPDFError(f"Empty PDF detected: '{filename}' has a file size of 0 bytes.")
            
        dest_path = os.path.join(upload_dir, filename)
        
        # Write file content
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        saved_paths.append(dest_path)
        
    return saved_paths

def process_uploaded_documents(file_paths):
    """
    Loads PDF documents using PyPDFLoader, validates them, and splits them into chunks.
    Ensures that source metadata is preserved.
    """
    all_chunks = []
    
    for path in file_paths:
        filename = os.path.basename(path)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
        except Exception as e:
            raise CorruptedPDFError(f"Corrupted PDF detected: Failed to parse and load '{filename}'. Error: {str(e)}")
            
        if not docs:
            raise EmptyPDFError(f"Empty PDF detected: '{filename}' contains no pages.")
            
        # Check if all pages are completely empty/blank text
        if all(not doc.page_content.strip() for doc in docs):
            raise EmptyPDFError(f"Empty PDF detected: '{filename}' contains no extractable or readable text.")
            
        try:
            text_splitter = CharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=500
            )
            chunks = text_splitter.split_documents(docs)
            # Ensure each chunk has source file name preserved in its metadata
            for chunk in chunks:
                if 'source' not in chunk.metadata:
                    chunk.metadata['source'] = path
            all_chunks.extend(chunks)
        except Exception as e:
            raise RuntimeError(f"Error splitting text in '{filename}': {str(e)}")
            
    return all_chunks

def add_documents_to_vectorstore(new_chunks):
    """
    Appends the document chunks to Chroma and persists the changes.
    Maps exception to EmbeddingFailureError or ChromaDBFailureError.
    """
    if not new_chunks:
        return
        
    try:
        # Add to the existing vector store
        st.session_state.vectorstore.add_documents(new_chunks)
        
        # Persist ChromaDB
        if hasattr(st.session_state.vectorstore, "persist"):
            st.session_state.vectorstore.persist()
    except Exception as e:
        err_msg = str(e).lower()
        if "embed" in err_msg or "huggingface" in err_msg or "rate limit" in err_msg or "request" in err_msg:
            raise EmbeddingFailureError(f"Embedding failure: Failed to generate vectors using HuggingFace. Details: {str(e)}")
        else:
            raise ChromaDBFailureError(f"ChromaDB storage failure: Failed to save documents to database. Details: {str(e)}")

def refresh_retrieval_chain():
    """
    Rebuilds the retrieval chain and conversational interface.
    Preserves chat history in conversational chain memory.
    """
    try:
        old_chain = st.session_state.get("conversationsal_chain")
        new_chain = chat_chain(st.session_state.vectorstore)
        
        # Copy message history to preserve context
        if old_chain and hasattr(old_chain, "memory") and old_chain.memory:
            try:
                new_chain.memory.chat_memory.messages = old_chain.memory.chat_memory.messages
            except Exception:
                pass
                
        st.session_state.conversationsal_chain = new_chain
    except Exception as e:
        raise RuntimeError(f"Failed to refresh conversation memory/chain: {str(e)}")

# Load Custom CSS & JavaScript helpers
def inject_custom_assets():
    css_path = os.path.join(working_dir, "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    # Inject background orbs and JS toast + interactions
    st.markdown(
        """
        <div class="orb-container">
            <div class="orb orb-purple"></div>
            <div class="orb orb-blue"></div>
            <div class="orb orb-pink"></div>
        </div>
        <div id="custom-toast" class="custom-toast"></div>
        
        <script>
        function showToast(message) {
            const toast = document.getElementById('custom-toast');
            if (toast) {
                toast.textContent = message;
                toast.className = 'custom-toast show';
                setTimeout(function() {
                    toast.className = 'custom-toast';
                }, 3000);
            }
        }
        
        function copyToClipboard(textId) {
            const textElement = document.getElementById(textId);
            if (textElement) {
                const text = textElement.getAttribute('data-raw-text') || textElement.innerText;
                navigator.clipboard.writeText(text).then(function() {
                    showToast('Copied response to clipboard!');
                }).catch(function() {
                    showToast('Failed to copy text.');
                });
            }
        }
        
        function handleFeedback(btn, type) {
            btn.classList.toggle('active');
            const row = btn.closest('.feedback-actions');
            if (row) {
                const buttons = row.querySelectorAll('.action-btn');
                buttons.forEach(b => {
                    if (b !== btn) b.classList.remove('active');
                });
            }
            showToast('Feedback recorded! Thank you.');
        }
        </script>
        """,
        unsafe_allow_html=True
    )

def get_pdf_files():
    data_dir = os.path.join(working_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

def format_message_content(content):
    # Parse markdown to HTML
    html_content = markdown.markdown(
        content,
        extensions=['fenced_code', 'tables']
    )
    return html_content

def render_upload_status(active_step):
    steps = [
        ("Uploading...", 0),
        ("Processing document...", 1),
        ("Generating embeddings...", 2),
        ("Ready to chat.", 3)
    ]
    html = '<div class="upload-status-container">'
    for label, idx in steps:
        if idx < active_step:
            status_class = "upload-step completed"
        elif idx == active_step:
            status_class = "upload-step active"
        else:
            status_class = "upload-step"
        html += f'<div class="{status_class}"><span class="upload-step-dot"></span><span>{label}</span></div>'
    html += '</div>'
    return html

def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 25px; padding-top: 10px;">
            <span style="font-size: 42px; filter: drop-shadow(0 0 10px rgba(168, 85, 247, 0.4));">📚</span>
            <h2 style="margin: 10px 0 0 0; font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 800; background: linear-gradient(135deg, #a5b4fc, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AskDocs AI</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.sidebar.markdown('<div class="sidebar-section-title">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        to_process = [
            f for f in uploaded_files 
            if f.name not in st.session_state.get("indexed_documents", [])
            and f.name not in st.session_state.get("failed_uploads", set())
        ]
        
        if to_process:
            status_placeholder = st.sidebar.empty()
            try:
                # Step 0: Uploading...
                status_placeholder.markdown(render_upload_status(0), unsafe_allow_html=True)
                saved_paths = save_uploaded_files(to_process)
                
                # Step 1: Processing document...
                status_placeholder.markdown(render_upload_status(1), unsafe_allow_html=True)
                new_chunks = process_uploaded_documents(saved_paths)
                
                # Step 2: Generating embeddings...
                status_placeholder.markdown(render_upload_status(2), unsafe_allow_html=True)
                add_documents_to_vectorstore(new_chunks)
                refresh_retrieval_chain()
                
                # Update states
                for f in to_process:
                    if f.name not in st.session_state.indexed_documents:
                        st.session_state.indexed_documents.append(f.name)
                    st.session_state.failed_uploads.discard(f.name)
                
                # Step 3: Ready to chat.
                status_placeholder.markdown(render_upload_status(3), unsafe_allow_html=True)
                import time
                time.sleep(1.5)
                status_placeholder.empty()
            except DuplicateUploadError:
                status_placeholder.empty()
                st.sidebar.error("This document is already indexed.")
                for f in to_process:
                    st.session_state.failed_uploads.add(f.name)
            except EmptyPDFError:
                status_placeholder.empty()
                st.sidebar.error("The uploaded PDF is empty or has no readable text.")
                for f in to_process:
                    st.session_state.failed_uploads.add(f.name)
            except CorruptedPDFError:
                status_placeholder.empty()
                st.sidebar.error("Failed to parse the PDF. It might be corrupted.")
                for f in to_process:
                    st.session_state.failed_uploads.add(f.name)
            except (EmbeddingFailureError, ChromaDBFailureError):
                status_placeholder.empty()
                st.sidebar.error("Database error occurred during vector generation.")
                for f in to_process:
                    st.session_state.failed_uploads.add(f.name)
            except Exception:
                status_placeholder.empty()
                st.sidebar.error("Failed to process uploaded documents.")
                for f in to_process:
                    st.session_state.failed_uploads.add(f.name)
                
    st.sidebar.markdown('<div class="sidebar-section-title">Uploaded Documents</div>', unsafe_allow_html=True)
    if "indexed_documents" in st.session_state and st.session_state.indexed_documents:
        for file in st.session_state.indexed_documents:
            col1, col2 = st.sidebar.columns([0.83, 0.17])
            with col1:
                file_html = f"""
                <div class="file-list-item" style="margin-bottom: 0;">
                    <span class="file-icon">📄</span>
                    <span class="file-name" title="{file}">{file}</span>
                </div>
                """
                st.markdown(file_html, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{file}", help=f"Remove {file}"):
                    if delete_document_from_vectorstore(st.session_state.vectorstore, file):
                        st.session_state.indexed_documents.remove(file)
                        st.session_state.failed_uploads.discard(file)
                        refresh_retrieval_chain()
                        st.rerun()
    else:
        st.sidebar.markdown('<div style="font-size: 12px; color: #64748b; font-style: italic; margin-bottom: 15px; padding: 0 5px;">No PDFs uploaded yet.</div>', unsafe_allow_html=True)
        
    st.sidebar.markdown('<div class="sidebar-section-title">Actions</div>', unsafe_allow_html=True)
    
    if st.sidebar.button("New Chat", key="new_chat_btn", use_container_width=True):
        st.session_state.chat_history = []
        if "conversationsal_chain" in st.session_state:
            st.session_state.conversationsal_chain = chat_chain(st.session_state.vectorstore)
        st.rerun()
        
    if st.sidebar.button("Clear Chat", key="clear_chat_btn", use_container_width=True):
        st.session_state.chat_history = []
        if "conversationsal_chain" in st.session_state:
            st.session_state.conversationsal_chain = chat_chain(st.session_state.vectorstore)
        st.rerun()

    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
            Powered by AskDocs AI
        </div>
        """,
        unsafe_allow_html=True
    )

def render_header():
    header_html = """
    <div class="main-header">
        <h1>📚 Multi-Document AI Assistant</h1>
        <p>Chat with your PDFs using Generative AI</p>
    </div>
    <div class="glowing-divider"></div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_welcome_screen():
    welcome_html = """
    <div class="welcome-container">
        <h1 class="welcome-title">Multi-Document AI Assistant</h1>
        <p class="welcome-subtitle">Upload PDFs and ask questions in natural language.</p>
        <div class="welcome-cards-grid">
            <div class="welcome-card">
                <span class="welcome-card-icon">📄</span>
                <div class="welcome-card-title">Upload Documents</div>
                <div class="welcome-card-desc">Upload multiple PDF files to index them into the vector database.</div>
            </div>
            <div class="welcome-card">
                <span class="welcome-card-icon">💬</span>
                <div class="welcome-card-title">Ask Questions</div>
                <div class="welcome-card-desc">Interact with the chatbot to ask anything about your indexed documents.</div>
            </div>
            <div class="welcome-card">
                <span class="welcome-card-icon">⚡</span>
                <div class="welcome-card-title">Get Instant AI Answers</div>
                <div class="welcome-card-desc">Get responses generated with retrieved context and relevant sources.</div>
            </div>
        </div>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)

def render_chat_messages():
    chat_html = '<div class="chat-container">'
    
    for idx, msg in enumerate(st.session_state.chat_history):
        role = msg["role"]
        content = msg["content"]
        timestamp = msg.get("timestamp", "")
        
        if role == "user":
            chat_html += f"""<div class="message-row user-row">
<div class="message-bubble user-bubble">
<div class="message-text">{html.escape(content)}</div>
</div>
<div class="message-timestamp user-timestamp">{timestamp}</div>
</div>"""
        else:
            html_content = format_message_content(content)
            escaped_raw = html.escape(content)
            
            # Sources rendering
            sources_html = ""
            sources = msg.get("sources", [])
            if sources:
                sources_html += '<div class="source-section"><div class="source-title">Sources</div><div class="source-grid">'
                for src in sources:
                    name = src["name"]
                    page = src.get("page", None)
                    page_str = f"Page {page + 1}" if page is not None else "Page N/A"
                    sources_html += f"""<div class="source-card">
<span class="source-card-icon">📄</span>
<div class="source-card-details">
<span class="source-card-name" title="{name}">{name}</span>
<span class="source-card-page">{page_str}</span>
</div>
</div>"""
                sources_html += '</div></div>'
            
            chat_html += f"""<div class="message-row assistant-row">
<div class="message-bubble assistant-bubble">
<div class="message-text" id="msg-{idx}" data-raw-text="{escaped_raw}">{html_content}</div>
{sources_html}
<div class="feedback-actions">
<button onclick="copyToClipboard('msg-{idx}')" class="action-btn" title="Copy response">
<svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> Copy
</button>
<button onclick="handleFeedback(this, 'up')" class="action-btn" title="Thumbs up">
<svg viewBox="0 0 24 24"><path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-1.91l-.01-.01L23 10z"/></svg> Helpful
</button>
<button onclick="handleFeedback(this, 'down')" class="action-btn" title="Thumbs down">
<svg viewBox="0 0 24 24"><path d="M19 15h4V3h-4v12zm-4 0c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73V4c0-1.1-.9-2-2-2H9c-.83 0-1.54.5-1.84 1.22L4.14 10.27c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L12.83 23l6.59-6.59c.37-.36.58-.86.58-1.41z"/></svg> Unhelpful
</button>
</div>
</div>
<div class="message-timestamp assistant-timestamp">{timestamp}</div>
</div>"""
            
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

def render_input_box():
    user_input = st.chat_input("Ask anything about your documents...")
    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        st.rerun()

# ----------------- Main Flow -----------------

# Page Configuration
st.set_page_config(
    page_title="Multi-Document AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = get_indexed_documents(st.session_state.vectorstore)

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "failed_uploads" not in st.session_state:
    st.session_state.failed_uploads = set()

if "conversationsal_chain" not in st.session_state:
    st.session_state.conversationsal_chain = chat_chain(
        st.session_state.vectorstore
    )

# Inject CSS and JavaScript assets
inject_custom_assets()

# Sidebar Components
render_sidebar()

# Main Flow Conditional Layout
if not st.session_state.chat_history:
    render_welcome_screen()
else:
    render_header()
    render_chat_messages()

# Handle Assistant Response Generation (when the last message is from user)
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    # Show typing indicator
    with st.empty():
        st.markdown(
            """<div class="chat-container">
<div class="message-row assistant-row">
<div class="message-bubble assistant-bubble">
<div class="typing-indicator">
<span></span>
<span></span>
<span></span>
</div>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )
        
        user_input = st.session_state.chat_history[-1]["content"]
        
        # Invoke RAG chain
        response = st.session_state.conversationsal_chain(
            {"question": user_input}
        )
        
        assistant_response = response["answer"]
        source_docs = response.get("source_documents", [])
        
        sources = []
        for doc in source_docs:
            source_path = doc.metadata.get("source", "Unknown")
            source_name = os.path.basename(source_path)
            page = doc.metadata.get("page", None)
            
            source_info = {"name": source_name, "page": page}
            if source_info not in sources:
                sources.append(source_info)
                
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response,
            "sources": sources,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        
    st.rerun()

# Input box at bottom
render_input_box()