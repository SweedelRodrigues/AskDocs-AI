try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

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
import web_search
import logging
logger = logging.getLogger(__name__)

import hashlib
import textwrap
import base64

def calculate_file_hash(filepath):
    """
    Calculates the SHA-256 hash of a file.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return f"hash_error_{os.path.basename(filepath)}"

def load_summaries_cache():
    """
    Loads the summaries cache from data/summaries_cache.json.
    """
    cache_path = os.path.join(working_dir, "data", "summaries_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_summaries_cache(cache):
    """
    Saves the summaries cache to data/summaries_cache.json.
    """
    data_dir = os.path.join(working_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cache_path = os.path.join(data_dir, "summaries_cache.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.sidebar.error(f"Error saving summary cache: {str(e)}")

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

# Read config.json unconditionally to ensure config_data is defined
config_data = {}
config_path = os.path.join(working_dir, "config.json")
if os.path.exists(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:
        pass

# Read Groq API Key from environment variable first, then fallback to config.json
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = config_data.get("GROQ_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY


def extract_document_metadata(docs, filepath):
    """
    Extracts metadata from the loaded document list.
    """
    word_count = sum(len(page.page_content.split()) for page in docs)
    total_pages = len(docs)
    reading_time = max(1, round(word_count / 200)) # 200 wpm
    upload_time = datetime.now().strftime("%B %d, %Y, %I:%M %p")
    return {
        "filename": os.path.basename(filepath),
        "total_pages": total_pages,
        "word_count": word_count,
        "reading_time": reading_time,
        "upload_time": upload_time
    }

def generate_document_analysis(doc_text):
    """
    Invokes Groq Llama 3.3 to perform a single-pass extraction of all document insights in structured JSON format.
    """
    truncated_text = doc_text[:40000]
    
    prompt = f"""You are an expert AI research and document analysis assistant.
Analyze the following document content and extract detailed structured insights.

Your response MUST be a valid JSON object. Do NOT include any markdown code blocks, triple backticks (```json), or leading/trailing conversational text. Only output the raw JSON object string.

The JSON object MUST contain the following keys exactly:
1. "quick_summary": A high-level, executive summary of 2-3 sentences.
2. "detailed_summary": A thorough summary explaining key details, methods, and insights (between 150 and 250 words).
3. "purpose": A brief explanation of the document's main purpose or objective.
4. "topics": A list of 5 to 10 key topics, concepts, or keywords.
5. "takeaways": A list of 3 to 5 key takeaway bullet points highlighting main ideas.
6. "intended_audience": A short description of the target/intended audience.
7. "difficulty_level": The reading difficulty level ("Beginner", "Intermediate", or "Advanced").
8. "suggested_questions": Exactly 5 intelligent questions that a user might ask about this document's content.
9. "conclusion": A single-sentence final conclusion of the document.

Document Content:
{truncated_text}

JSON Output:"""

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    try:
        response = llm.invoke(prompt)
        raw_content = response.content.strip()
        
        # Clean JSON blocks if LLM outputted them despite instructions
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        raw_content = raw_content.strip()
        
        data = json.loads(raw_content)
        
        # Ensure all required keys exist
        required_keys = ["quick_summary", "detailed_summary", "purpose", "topics", "takeaways", "intended_audience", "difficulty_level", "suggested_questions", "conclusion"]
        for key in required_keys:
            if key not in data:
                if key in ["topics", "takeaways", "suggested_questions"]:
                    data[key] = []
                else:
                    data[key] = "N/A"
        return data
    except Exception as e:
        # Fallback summary details in case of parse/API error
        return {
            "quick_summary": "Summary generation encountered an error or timed out.",
            "detailed_summary": f"Could not generate detailed summary. Error: {str(e)}",
            "purpose": "N/A",
            "topics": ["Error"],
            "takeaways": ["Failed to extract takeaways due to an API error."],
            "intended_audience": "N/A",
            "difficulty_level": "N/A",
            "suggested_questions": [
                "What is this document about?",
                "What are the main findings?",
                "What is the methodology?",
                "What are the conclusions?",
                "Can you explain the main ideas?"
            ],
            "conclusion": "N/A"
        }

def generate_document_summary(doc_text):
    analysis = generate_document_analysis(doc_text)
    return analysis.get("detailed_summary", "")

def extract_key_topics(doc_text):
    analysis = generate_document_analysis(doc_text)
    return analysis.get("topics", [])

def generate_key_takeaways(doc_text):
    analysis = generate_document_analysis(doc_text)
    return analysis.get("takeaways", [])

def cache_summary(filename, file_hash, summary_data):
    cache = load_summaries_cache()
    cache[file_hash] = {
        "filename": filename,
        "quick_summary": summary_data.get("quick_summary", ""),
        "detailed_summary": summary_data.get("detailed_summary", ""),
        "purpose": summary_data.get("purpose", ""),
        "topics": summary_data.get("topics", []),
        "takeaways": summary_data.get("takeaways", []),
        "intended_audience": summary_data.get("intended_audience", ""),
        "difficulty_level": summary_data.get("difficulty_level", ""),
        "suggested_questions": summary_data.get("suggested_questions", []),
        "conclusion": summary_data.get("conclusion", ""),
        "metadata": summary_data.get("metadata", {})
    }
    save_summaries_cache(cache)

def process_summary_for_file(filename, saved_path, status_placeholder):
    """
    Computes hash, checks cache, generates summary and metadata with visual progress status.
    """
    file_hash = calculate_file_hash(saved_path)
    cache = load_summaries_cache()
    
    if file_hash in cache:
        # Already cached
        st.session_state.selected_document = filename
        return
        
    def update_status(text):
        status_placeholder.markdown(
            textwrap.dedent(f"""
                <div class="summary-loading-card">
                    <div class="summary-loading-spinner"></div>
                    <div class="summary-loading-text">{text}</div>
                </div>
            """),
            unsafe_allow_html=True
        )
        
    import time
    
    # Show status: Analyzing document...
    update_status("Analyzing document...")
    time.sleep(0.8)
    
    # Show status: Extracting content...
    update_status("Extracting content...")
    try:
        loader = PyPDFLoader(saved_path)
        docs = loader.load()
        metadata = extract_document_metadata(docs, saved_path)
        doc_text = "\n".join([page.page_content for page in docs])
    except Exception as e:
        st.sidebar.error(f"Error loading document '{filename}': {str(e)}")
        return
        
    time.sleep(0.8)
    
    # Show status: Generating AI summary...
    update_status("Generating AI summary...")
    analysis = generate_document_analysis(doc_text)
    
    # Embed metadata into analysis for caching
    analysis["metadata"] = metadata
    
    # Cache it
    cache_summary(filename, file_hash, analysis)
    
    # Show status: Done!
    update_status("Done!")
    time.sleep(1.0)
    status_placeholder.empty()
    
    # Select document automatically
    st.session_state.selected_document = filename


def get_cached_summary_by_filename(filename):
    """
    Finds the cached summary data matching the given filename.
    """
    cache = load_summaries_cache()
    for file_hash, data in cache.items():
        if data.get("filename") == filename:
            return data
    return None

@st.cache_data
def render_pdf_page_as_png(pdf_path, page_num, highlight_text=None):
    """
    Renders a single PDF page as PNG bytes, optionally highlighting a specific text chunk.
    """
    import fitz
    import re
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    
    if highlight_text:
        # Normalize whitespace in highlight_text
        norm_text = re.sub(r'\s+', ' ', highlight_text.strip())
        
        # 1. Try search with normalized whitespace
        text_instances = page.search_for(norm_text)
        
        # 2. Try raw chunk search
        if not text_instances:
            text_instances = page.search_for(highlight_text)
            
        # 3. Try splitting the chunk into sentences/phrases if the exact full block is not found
        if not text_instances:
            phrases = [p.strip() for p in re.split(r'[.\n!?;]+', highlight_text) if len(p.strip()) > 15]
            for phrase in phrases:
                norm_phrase = re.sub(r'\s+', ' ', phrase)
                insts = page.search_for(norm_phrase)
                if not insts:
                    insts = page.search_for(phrase)
                text_instances.extend(insts)
                
        # 4. Try searching for the first 30 chars as fallback
        if not text_instances and len(norm_text) > 30:
            text_instances = page.search_for(norm_text[:30])
            
        # Draw highlight annotations on all matching rects
        for rect in text_instances:
            page.add_highlight_annot(rect)
            
    # Render page to a clean PNG image at 150 DPI for rich premium readability
    pix = page.get_pixmap(dpi=150)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes

def render_pdf_preview(filename):
    """
    Renders a PDF page-by-page as high-quality PNGs with visual text highlighting
    for retrieved citation chunks. Fallback is a direct download button.
    """
    upload_dir = os.path.join(working_dir, "uploaded_docs")
    upload_path = os.path.join(upload_dir, filename)
    data_dir = os.path.join(working_dir, "data")
    data_path = os.path.join(data_dir, filename)
    path_to_use = upload_path if os.path.exists(upload_path) else (data_path if os.path.exists(data_path) else None)
    
    if not path_to_use:
        st.error(f"❌ Error: The document '{filename}' could not be located in the uploads directory.")
        return
        
    if not os.path.exists(path_to_use):
        st.error(f"❌ Error: File not found at '{path_to_use}'.")
        return
        
    try:
        import fitz
        doc = fitz.open(path_to_use)
        total_pages = doc.page_count
        doc.close()
    except Exception as e:
        st.error(f"❌ Error loading PDF file: {str(e)}")
        return
        
    # Ensure session state variables for active page and selected document are initialized
    if "active_page" not in st.session_state or st.session_state.get("active_page_doc") != filename:
        st.session_state.active_page = 0
        st.session_state.active_page_doc = filename
        
    active_page = max(0, min(st.session_state.active_page, total_pages - 1))
    st.session_state.active_page = active_page
    
    # Render active text highlight if active
    highlight_text = None
    if "active_text" in st.session_state and st.session_state.active_text:
        highlight_text = st.session_state.active_text
        st.markdown(
            f"""
            <div style="background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <span style="color: #a855f7; font-weight: 600;">💡 Active RAG Citation Highlight:</span>
                <p style="font-size: 12.2px; color: #94a3b8; margin: 6px 0 10px 0; font-style: italic; line-height: 1.4;">"{highlight_text[:180]}..."</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Clear Highlight", key="clear_hl_btn", use_container_width=True):
            st.session_state.active_text = ""
            st.rerun()
            
    # Page Navigation Controls Bar
    col_prev, col_num, col_next = st.columns([0.25, 0.5, 0.25])
    with col_prev:
        if st.button("◀️ Prev Page", key="prev_pg_btn", disabled=(active_page == 0), use_container_width=True):
            st.session_state.active_page = active_page - 1
            st.rerun()
    with col_num:
        st.markdown(f"<div style='text-align: center; font-weight: 600; padding-top: 6px; color: #a855f7; font-size: 14px;'>Page {active_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
    with col_next:
        if st.button("Next Page ▶️", key="next_pg_btn", disabled=(active_page == total_pages - 1), use_container_width=True):
            st.session_state.active_page = active_page + 1
            st.rerun()

    # Display a subtle loading animation / spinner while page loads
    with st.spinner("Rendering PDF page..."):
        try:
            img_bytes = render_pdf_page_as_png(path_to_use, active_page, highlight_text)
            st.image(img_bytes, use_column_width=True)
        except Exception as e:
            st.error(f"⚠️ Page rendering failed: {str(e)}")
            try:
                with open(path_to_use, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📥 Download PDF Document (Fallback)",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception:
                pass

def render_document_metadata(metadata):
    """
    Renders document metadata metrics inside Streamlit columns.
    """
    pages = metadata.get("total_pages", "N/A")
    words = metadata.get("word_count", "N/A")
    reading_time = metadata.get("reading_time", "N/A")
    upload_time = metadata.get("upload_time", "N/A")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Pages", value=pages)
    with col2:
        st.metric(label="Reading Time", value=f"{reading_time} min")
    with col3:
        if isinstance(words, int):
            st.metric(label="Word Count", value=f"{words:,}")
        else:
            st.metric(label="Word Count", value=words)
    with col4:
        st.metric(label="Uploaded", value=upload_time)

def render_summary(quick_summary, detailed_summary):
    """
    Renders the quick summary text and detailed summary inside a beautiful border card.
    """
    with st.container(border=True):
        st.markdown(f"**Quick Summary:**  \n{quick_summary}")
        show_details = st.toggle("🔍 Show Detailed AI Summary", value=False)
        if show_details:
            st.divider()
            st.markdown(detailed_summary)

def render_key_topics(topics):
    """
    Renders keywords/topics as chips using clean inline HTML tags.
    """
    if not topics:
        st.caption("No topics extracted.")
        return
    chips_html = "".join([f'<span class="topic-chip">{html.escape(t)}</span>' for t in topics])
    st.markdown(f'<div class="topics-container">{chips_html}</div>', unsafe_allow_html=True)

def render_takeaways(takeaways, conclusion):
    """
    Renders bullet points of takeaways and a formatted conclusion.
    """
    if not takeaways:
        st.caption("No takeaways generated.")
    else:
        for t in takeaways:
            st.markdown(f"• {t}")
    if conclusion and conclusion != "N/A":
        st.markdown(f"*Conclusion: {conclusion}*")

def render_suggested_questions(questions):
    """
    Renders suggested questions as a row of pill buttons.
    """
    if not questions:
        return
    cols = st.columns(len(questions))
    for idx, q in enumerate(questions):
        with cols[idx]:
            if st.button(q, key=f"q_chip_{idx}", use_container_width=True):
                # Append user prompt and trigger immediate chatbot run
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": q,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                st.rerun()

def render_document_dashboard(filename):
    """
    Renders the entire Document Insights dashboard. It handles state expansion,
    metadata displaying, previews, and key insights widgets.
    """
    # Track document change to force expand
    if st.session_state.get("prev_selected_document") != filename:
        st.session_state.prev_selected_document = filename
        st.session_state.expand_insights = True
    else:
        # Don't force reset on standard reruns
        if "expand_insights" not in st.session_state:
            st.session_state.expand_insights = True

    # Get cached data
    summary_data = get_cached_summary_by_filename(filename)
    if not summary_data:
        # Generate on-the-fly fallback
        upload_path = os.path.join(working_dir, "uploaded_docs", filename)
        data_path = os.path.join(working_dir, "data", filename)
        path_to_use = upload_path if os.path.exists(upload_path) else (data_path if os.path.exists(data_path) else None)
        if path_to_use:
            status_placeholder = st.empty()
            process_summary_for_file(filename, path_to_use, status_placeholder)
            summary_data = get_cached_summary_by_filename(filename)
            
    if not summary_data:
        st.warning(f"Could not load insights for '{filename}'.")
        return

    # Render inside a collapsible expander
    is_expanded = st.session_state.get("expand_insights", True)
    
    with st.expander(f"📄 AI Document Insights: {filename}", expanded=is_expanded):
        # Once expanded, set to False so subsequent typing reruns don't force open it if closed manually
        st.session_state.expand_insights = False
        
        # 1. Metadata
        st.markdown("##### Metadata")
        with st.container(border=True):
            render_document_metadata(summary_data.get("metadata", {}))
        
        # 2. PDF Preview Card
        # Expand preview if we have an active citation highlighted
        preview_expanded = False
        if "active_text" in st.session_state and st.session_state.active_text:
            preview_expanded = True
            
        show_preview = st.toggle("👁️ Show PDF Document Preview", value=preview_expanded)
        if show_preview:
            st.markdown('<div id="pdf-preview-anchor"></div>', unsafe_allow_html=True)
            with st.container(border=True):
                render_pdf_preview(filename)
            
            if "active_text" in st.session_state and st.session_state.active_text:
                st.markdown(
                    """
                    <script>
                    setTimeout(function() {
                        const anchor = document.getElementById("pdf-preview-anchor");
                        if (anchor) {
                            anchor.scrollIntoView({ behavior: "smooth", block: "start" });
                        }
                    }, 300);
                    </script>
                    """,
                    unsafe_allow_html=True
                )
            
        st.divider()
        
        # 3. Summary
        st.markdown("##### Executive Summary")
        render_summary(summary_data.get("quick_summary", ""), summary_data.get("detailed_summary", ""))
        
        st.divider()
        
        # 4. Context & Takeaways Columns
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.markdown("##### 🎯 Document Context")
                st.markdown(f"**Purpose:** {summary_data.get('purpose', 'N/A')}")
                st.markdown(f"**Intended Audience:** {summary_data.get('intended_audience', 'N/A')}")
                st.markdown(f"**Difficulty Level:** {summary_data.get('difficulty_level', 'N/A')}")
                
                st.markdown("##### 🏷️ Key Topics")
                render_key_topics(summary_data.get("topics", []))
            
        with col2:
            with st.container(border=True):
                st.markdown("##### 📝 Key Takeaways")
                render_takeaways(summary_data.get("takeaways", []), summary_data.get("conclusion", ""))
            
        st.divider()
        
        # 5. Suggested Questions
        st.markdown("##### 💡 Suggested Questions")
        with st.container(border=True):
            render_suggested_questions(summary_data.get("suggested_questions", []))
        
        st.write("") # Add spacing
        
        # Back to general chat button
        col_back, _ = st.columns([0.25, 0.75])
        with col_back:
            if st.button("⬅️ General Chat (Deselect File)", key="back_to_general_btn", use_container_width=True):
                st.session_state.selected_document = None
                st.rerun()





def is_summary_query(query: str) -> bool:
    """
    Checks if the user prompt is a request for a document summary or highlights.
    """
    query_lower = query.strip().lower().rstrip("?.!")
    keywords = [
        "summarize this",
        "summarize the document",
        "summarize document",
        "give me a summary",
        "give me the summary",
        "explain this document",
        "explain the document",
        "what is this document about",
        "what is the document about",
        "what is this pdf about",
        "what is the pdf about",
        "give me key takeaways",
        "give key takeaways",
        "what are the key takeaways",
        "what are the important points",
        "explain this pdf",
        "explain the pdf",
        "give summary",
        "get summary",
        "explain pdf",
        "summarize",
        "executive summary",
        "highlights",
        "important findings",
        "purpose of this document",
        "purpose of the document",
        "intended audience",
        "main conclusions",
        "explain this like i'm a beginner",
        "explain this like a beginner"
    ]
    if any(k in query_lower for k in keywords):
        return True
        
    import re
    if re.search(r"\b(summarize|summary|explain|takeaway|takeaways|important|points|purpose|audience|conclusions|highlights)\b", query_lower):
        if re.search(r"\b(document|pdf|file|paper|text|this|about|about\s+this)\b", query_lower) or query_lower in ["summarize", "explain"]:
            return True
    return False

def map_query_to_cache_field(query: str):
    """
    Maps a natural language query to a specific cached summary field.
    """
    query_lower = query.lower().strip().rstrip("?.!")
    
    # Purpose
    if any(p in query_lower for p in ["purpose of this document", "purpose of the document", "document's purpose", "what is the purpose"]):
        return "purpose", "Purpose"
        
    # Audience
    if any(a in query_lower for a in ["intended audience", "target audience", "who is the audience", "who is this document for"]):
        return "intended_audience", "Intended Audience"
        
    # Conclusion
    if any(c in query_lower for c in ["main conclusions", "document's conclusion", "what is the conclusion", "conclusions"]):
        return "conclusion", "Conclusion"
        
    # Takeaways / Key points
    if any(t in query_lower for t in ["key points", "important findings", "key takeaways", "takeaways", "highlights", "important points"]):
        return "takeaways", "Key Takeaways"
        
    # Executive Summary / Beginner
    if "beginner" in query_lower or "explain this like i'm a beginner" in query_lower:
        return "quick_summary", "Executive Summary (Beginner-friendly)"
        
    if "executive summary" in query_lower or "quick summary" in query_lower or "short summary" in query_lower:
        return "quick_summary", "Executive Summary"

    return "detailed_summary", "Detailed AI Summary"


def get_target_document_for_query(query: str) -> str:
    """
    Determines which document in the vector store is the subject of the summary request.
    """
    docs = st.session_state.get("indexed_documents", [])
    if not docs:
        return None
    if len(docs) == 1:
        return docs[0]
        
    # Check if a filename is mentioned in the query (case insensitive)
    query_lower = query.lower()
    for doc in docs:
        if doc.lower() in query_lower:
            return doc
        name_no_ext = os.path.splitext(doc)[0].lower()
        if name_no_ext in query_lower:
            return doc
        name_clean = name_no_ext.replace("_", " ").replace("-", " ")
        if name_clean in query_lower:
            return doc
            
    # Check if a document is selected in the sidebar
    selected = st.session_state.get("selected_document")
    if selected in docs:
        return selected
        
    # Fallback to the most recently uploaded document (last in list)
    return docs[-1]








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

        function setCitation(doc, page, msgIdx, srcIdx) {
            console.log("setCitation called with:", doc, page, msgIdx, srcIdx);
            let targetInput = null;
            
            // 1. Try local document query first
            try {
                targetInput = document.querySelector('input[aria-label="Active Citation Helper"]') ||
                              document.querySelector('input[placeholder="Active Citation Helper"]');
                if (targetInput) console.log("Found targetInput locally via querySelector");
            } catch (e) {
                console.error("Local querySelector error:", e);
            }
            
            // 2. Try local wrappers loop fallback
            if (!targetInput) {
                try {
                    const localWrappers = document.querySelectorAll('[data-testid="stTextInput"]');
                    for (const wrapper of localWrappers) {
                        const label = wrapper.querySelector('label');
                        if (label && (label.textContent || label.innerText || "").trim() === "Active Citation Helper") {
                            targetInput = wrapper.querySelector('input');
                            if (targetInput) {
                                console.log("Found targetInput locally via wrappers loop");
                                break;
                            }
                        }
                    }
                } catch (e) {
                    console.error("Local wrapper loop error:", e);
                }
            }
            
            // 3. Fallback to window.parent context (wrapped in try-catch for cross-origin compliance on Streamlit Cloud)
            if (!targetInput) {
                try {
                    targetInput = window.parent.document.querySelector('input[aria-label="Active Citation Helper"]') ||
                                  window.parent.document.querySelector('input[placeholder="Active Citation Helper"]');
                    if (targetInput) console.log("Found targetInput in parent via querySelector");
                } catch (e) {
                    console.warn("Parent querySelector access blocked by cross-origin policy:", e);
                }
            }
            
            // 4. Try parent wrappers loop fallback
            if (!targetInput) {
                try {
                    const wrappers = window.parent.document.querySelectorAll('[data-testid="stTextInput"]');
                    for (const wrapper of wrappers) {
                        const label = wrapper.querySelector('label');
                        if (label && (label.textContent || label.innerText || "").trim() === "Active Citation Helper") {
                            targetInput = wrapper.querySelector('input');
                            if (targetInput) {
                                console.log("Found targetInput in parent via wrappers loop");
                                break;
                            }
                        }
                    }
                } catch (e) {
                    console.warn("Parent wrapper loop access blocked by cross-origin policy:", e);
                }
            }
            
            if (targetInput) {
                const value = JSON.stringify({doc: doc, page: page, msgIdx: msgIdx, srcIdx: srcIdx, rand: Math.random()});
                console.log("Setting input value to:", value);
                
                try {
                    const nativeValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                    nativeValueSetter.call(targetInput, value);
                } catch (e) {
                    console.warn("Native value setter failed, falling back to direct assignment", e);
                    targetInput.value = value;
                }
                
                const tracker = targetInput._valueTracker;
                if (tracker) {
                    try {
                        tracker.setValue("");
                    } catch (e) {}
                }
                
                targetInput.dispatchEvent(new Event('input', { bubbles: true }));
                targetInput.dispatchEvent(new Event('change', { bubbles: true }));
                console.log("Events dispatched successfully");
            } else {
                console.error("Active Citation Helper input element not found in any scope!");
            }
        }

        // Global Event Delegation for custom click triggers (using Capture phase)
        document.addEventListener('click', function(event) {
            // Handle Copy button trigger
            const copyTrigger = event.target.closest('.copy-trigger');
            if (copyTrigger) {
                console.log("Copy clicked! Element:", copyTrigger);
                const targetId = copyTrigger.getAttribute('data-target');
                if (targetId) {
                    copyToClipboard(targetId);
                }
                return;
            }

            // Handle RAG Citation trigger
            const trigger = event.target.closest('.citation-trigger');
            if (trigger) {
                console.log("Citation clicked! Element:", trigger);
                const doc = trigger.getAttribute('data-doc');
                const page = parseInt(trigger.getAttribute('data-page'));
                const msgIdx = parseInt(trigger.getAttribute('data-msg-idx'));
                const srcIdx = parseInt(trigger.getAttribute('data-src-idx'));
                console.log("Extracted attributes:", doc, page, msgIdx, srcIdx);
                if (doc && !isNaN(page)) {
                    setCitation(doc, page, msgIdx, srcIdx);
                }
            }
        }, true);
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
                
                # Process automatic AI summary generation & caching for each uploaded file
                for filename, saved_path in zip([f.name for f in to_process], saved_paths):
                    process_summary_for_file(filename, saved_path, status_placeholder)

                # Step 3: Ready to chat.
                status_placeholder.markdown(render_upload_status(3), unsafe_allow_html=True)
                import time
                time.sleep(1.5)
                status_placeholder.empty()
                st.rerun()
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
                selected = st.session_state.get("selected_document") == file
                btn_label = f"✨ {file}" if selected else f"📄 {file}"
                if st.button(btn_label, key=f"doc_{file}", use_container_width=True):
                    st.session_state.selected_document = file
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{file}", help=f"Remove {file}"):
                    if delete_document_from_vectorstore(st.session_state.vectorstore, file):
                        st.session_state.indexed_documents.remove(file)
                        st.session_state.failed_uploads.discard(file)
                        if st.session_state.get("selected_document") == file:
                            st.session_state.selected_document = None
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
    header_html = textwrap.dedent(
        """
        <div class="main-header">
            <h1>📚 Multi-Document AI Assistant</h1>
            <p>Chat with your PDFs using Generative AI</p>
        </div>
        <div class="glowing-divider"></div>
        """
    )
    st.markdown(header_html, unsafe_allow_html=True)

def render_welcome_screen():
    st.markdown("# 📄 Welcome to AskDocs AI")
    st.markdown("##### Upload one or more PDF documents and start chatting with them using AI.")
    st.write("")
    
    # 6 features grid
    features = [
        ("✨ AI Document Summaries", "Get instant executive summaries, key topics, and key takeaways for any uploaded document."),
        ("🔍 Semantic Search", "Find exact details across all pages using semantic document retrieval techniques."),
        ("💬 Intelligent Q&A", "Engage in dialogue with documents. Ask questions naturally and get instant answers."),
        ("📚 Multi-Doc Support", "Upload multiple documents and cross-reference information seamlessly in one chat."),
        ("📌 Source Citations", "Every response lists the corresponding document and exact page citations."),
        ("💡 Suggested Questions", "Get 5 smart, auto-generated clickable question recommendations for every document.")
    ]
    
    col1, col2 = st.columns(2)
    for i, (title, desc) in enumerate(features):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.markdown(desc)
                
    st.write("")
    
    # Get Started Card using native Streamlit containers
    with st.container(border=True):
        st.markdown("### 📌 Get Started")
        st.markdown("Use the sidebar to upload one or more PDF documents.")
        st.markdown(
            """
            Once uploaded, the AI will automatically:
            * Generate an executive summary
            * Extract key topics
            * Generate key takeaways
            * Build the searchable knowledge base
            * Enable intelligent Q&A
            """
        )
        if st.button("Upload Documents", key="upload_docs_cta_btn", use_container_width=True):
            st.info("👈 Use the file uploader in the sidebar on the left to upload your PDF files.")

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
            
            # Source Badge rendering
            badge_html = ""
            source_type = msg.get("source_type", None)
            if source_type == "uploaded_docs":
                badge_html = '<div class="source-badge doc-badge">📄 Uploaded Documents</div>'
            elif source_type == "web":
                badge_html = '<div class="source-badge web-badge">🌐 Web</div>'
            elif source_type == "hybrid":
                badge_html = '<div class="source-badge hybrid-badge">📄 + 🌐 Hybrid</div>'
            
            # Sources rendering
            sources_html = ""
            sources = msg.get("sources", [])
            web_sources = msg.get("web_sources", [])
            
            if sources or web_sources:
                sources_html += '<div class="source-section"><div class="source-title">Sources</div><div class="source-grid">'
                import urllib.parse
                for s_idx, src in enumerate(sources):
                    name = src["name"]
                    page = src.get("page", None)
                    page_str = f"Page {page + 1}" if page is not None else "Page N/A"
                    safe_name = urllib.parse.quote(name)
                    
                    if page is not None:
                        escaped_name = html.escape(name)
                        sources_html += f"""<div class="source-card citation-trigger" data-doc="{escaped_name}" data-page="{page}" data-msg-idx="{idx}" data-src-idx="{s_idx}" style="cursor: pointer; text-decoration: none; color: inherit; display: flex; align-items: center; gap: 8px;">
<span class="source-card-icon">📄</span>
<div class="source-card-details">
<span class="source-card-name" title="{escaped_name}">{escaped_name}</span>
<span class="source-card-page" style="color: #a855f7;">{page_str} (Click to view)</span>
</div>
</div>"""
                    else:
                        sources_html += f"""<div class="source-card" style="display: flex; align-items: center; gap: 8px;">
<span class="source-card-icon">📄</span>
<div class="source-card-details">
<span class="source-card-name" title="{name}">{name}</span>
<span class="source-card-page">{page_str}</span>
</div>
</div>"""
                for wsrc in web_sources:
                    name = wsrc.get("name", "Web Source")
                    url = wsrc.get("url", "#")
                    snippet = wsrc.get("snippet", "")
                    short_snippet = (snippet[:150] + "...") if len(snippet) > 150 else snippet
                    sources_html += f"""<a href="{url}" target="_blank" class="source-card web-source-card" style="text-decoration: none; color: inherit; display: block; padding: 12px; height: auto; min-height: 80px;">
<div style="display: flex; align-items: flex-start; gap: 12px;">
<span class="source-card-icon" style="margin-top: 2px;">🌐</span>
<div class="source-card-details" style="flex: 1; overflow: hidden;">
<span class="source-card-name" title="{name}" style="font-weight: 600; color: #e2e8f0; display: block; text-overflow: ellipsis; white-space: nowrap; overflow: hidden; font-size: 13px;">{name}</span>
<span class="source-card-page" style="font-size: 10.5px; color: #a855f7; display: block; text-overflow: ellipsis; white-space: nowrap; overflow: hidden; margin-top: 2px;">{url}</span>
<span class="source-card-snippet" style="font-size: 11px; color: #94a3b8; display: block; margin-top: 6px; line-height: 1.4; white-space: normal; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">{html.escape(short_snippet)}</span>
</div>
</div>
</a>"""
                sources_html += '</div></div>'
            
            chat_html += f"""<div class="message-row assistant-row">
<div class="message-bubble assistant-bubble">
{badge_html}
<div class="message-text" id="msg-{idx}" data-raw-text="{escaped_raw}">{html_content}</div>
{sources_html}
                <div class="feedback-actions">
                    <button class="action-btn copy-trigger" data-target="msg-{idx}" title="Copy response">
                        <svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg> Copy
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

def get_selected_doc_path(filename):
    if not filename:
        return None
    upload_path = os.path.join(working_dir, "uploaded_docs", filename)
    data_path = os.path.join(working_dir, "data", filename)
    if os.path.exists(upload_path):
        return upload_path
    elif os.path.exists(data_path):
        return data_path
    return None

def get_chat_history_str(limit=5):
    history_msgs = []
    recent_history = st.session_state.chat_history[-limit:] if len(st.session_state.chat_history) > 0 else []
    for msg in recent_history:
        if msg == st.session_state.chat_history[-1] and msg["role"] == "user":
            continue
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        history_msgs.append(f"{role}: {content}")
    return "\n".join(history_msgs)

def generate_rag_answer_and_check_completeness(query, context_text, history_str=""):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    history_context = f"\nConversation History:\n{history_str}\n" if history_str else ""
    prompt = f"""You are an expert document assistant. You answer questions based on the provided document context.
Your goal is to answer the query using the context, taking the conversation history into account if relevant, and evaluate if the context was sufficient to answer the query completely.

Format your response as a valid JSON object. Do NOT include any markdown code blocks, triple backticks (```json), or leading/trailing conversational text. Only output the raw JSON object string.

The JSON object MUST contain the following keys exactly:
1. "can_answer": true if the context contains ANY information that is relevant to help construct an answer (even a partial or brief one). Set this to true as long as there is any topic or keyword overlap that helps address the user query.
2. "is_complete": true if the context contains enough information to answer the user query fully. Set this to false only if the user query explicitly asks for specific external details, facts, or questions that are completely absent from the context and require a web search. If the user query is simple and the context covers the core answer, set this to true.
3. "answer": A natural, direct, and detailed answer to the query based ONLY on the provided context. If can_answer is false, this should be empty or a brief statement of what is missing.
4. "missing_info": A brief description of what parts of the user's query are not addressed by the context, if applicable.

Context:
{context_text}
{history_context}
User Query:
{query}

JSON Output:"""

    try:
        response = llm.invoke(prompt)
        raw_content = response.content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        raw_content = raw_content.strip()
        
        data = json.loads(raw_content)
        return (
            data.get("answer", "").strip(),
            bool(data.get("is_complete")),
            bool(data.get("can_answer")),
            data.get("missing_info", "").strip()
        )
    except Exception as e:
        logger.error(f"Error checking completeness: {str(e)}")
        return "", False, False, ""

def generate_hybrid_answer(query, pdf_context, web_context, history_str=""):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    history_context = f"\nConversation History:\n{history_str}\n" if history_str else ""
    prompt = f"""You are an assistant that combines information from uploaded PDF documents and web search results.
Answer the user's query using both sources, taking the conversation history into account if relevant.
You MUST clearly separate the answer into two distinct sections as shown below:

📄 From Uploaded Documents
[Detailed answer based ONLY on the uploaded PDF document context. Do not include any web information here. If the PDFs don't mention a specific point, do not talk about it in this section.]

🌐 Additional Information from the Web
[Detailed answer based ONLY on the web search results, supplementing the PDF context to fully answer the query. Do not repeat what was already said in the PDF section, focus on the additional/missing details.]

If one of the sources doesn't contain any useful information, omit its section and only output the other section.
Do not mention source numbers or cite files in the web section. Keep the tone professional, direct, and helpful.

PDF Document Context:
{pdf_context}

Web Search Context:
{web_context}
{history_context}
User Query:
{query}

Answer:"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating hybrid answer: {str(e)}")
        return "Error generating hybrid answer."

def generate_web_answer(query, web_context, history_str=""):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    history_context = f"\nConversation History:\n{history_str}\n" if history_str else ""
    prompt = f"""You are a helpful assistant. Answer the user query using the provided web search results, taking the conversation history into account if relevant.
Be factual, direct, and concise. Do not hallucinate.
If the web search results do not contain enough information to answer the question, respond with exactly: "I couldn't find reliable information in your uploaded documents or trusted web sources."

Web Search Context:
{web_context}
{history_context}
User Query:
{query}

Answer:"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating web answer: {str(e)}")
        return "Error generating web answer."

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

if "selected_document" not in st.session_state:
    st.session_state.selected_document = None

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

# Hidden text input citation helper
st.markdown(
    """
    <style>
    div[data-testid="stTextInput"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
citation_helper = st.text_input("Active Citation Helper", placeholder="Active Citation Helper", key="active_citation_helper", value="")

if "last_processed_citation_rand" not in st.session_state:
    st.session_state.last_processed_citation_rand = None

if citation_helper:
    try:
        import json
        data = json.loads(citation_helper)
        rand_id = data.get("rand")
        if rand_id != st.session_state.last_processed_citation_rand:
            st.session_state.last_processed_citation_rand = rand_id
            doc_name = data.get("doc")
            page_num = data.get("page")
            m_idx = data.get("msgIdx")
            s_idx = data.get("srcIdx")
            
            if doc_name:
                st.session_state.selected_document = doc_name
                st.session_state.active_page = int(page_num)
                st.session_state.expand_insights = True
                
                # Fetch text chunk from message sources
                if 0 <= m_idx < len(st.session_state.chat_history):
                    msg = st.session_state.chat_history[m_idx]
                    if 0 <= s_idx < len(msg.get("sources", [])):
                        src = msg["sources"][s_idx]
                        st.session_state.active_text = src.get("text", "")
    except Exception as e:
        logger.error(f"Error parsing citation helper JSON: {str(e)}")

# Inject CSS and JavaScript assets
inject_custom_assets()

# Sidebar Components
render_sidebar()

# Main Flow Conditional Layout
if st.session_state.get("selected_document"):
    render_document_dashboard(st.session_state.selected_document)
    st.divider()
    st.markdown(f"### 💬 Chat with {st.session_state.selected_document}")
    if st.session_state.chat_history:
        render_chat_messages()
    else:
        st.info("Ask any question about this document below to start chatting.")
else:
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
        
        # Check if this is a summary query
        if is_summary_query(user_input):
            target_doc = get_target_document_for_query(user_input)
            if target_doc:
                summary_data = get_cached_summary_by_filename(target_doc)
                if not summary_data:
                    # Try to generate on the fly
                    upload_path = os.path.join(working_dir, "uploaded_docs", target_doc)
                    data_path = os.path.join(working_dir, "data", target_doc)
                    path_to_use = upload_path if os.path.exists(upload_path) else (data_path if os.path.exists(data_path) else None)
                    if path_to_use:
                        process_summary_for_file(target_doc, path_to_use, st.empty())
                        summary_data = get_cached_summary_by_filename(target_doc)
                
                if summary_data:
                    quick = summary_data.get("quick_summary", "")
                    detailed = summary_data.get("detailed_summary", "")
                    purpose = summary_data.get("purpose", "")
                    topics = summary_data.get("topics", [])
                    takeaways = summary_data.get("takeaways", [])
                    audience = summary_data.get("intended_audience", "")
                    difficulty = summary_data.get("difficulty_level", "")
                    conclusion = summary_data.get("conclusion", "")
                    
                    field, field_title = map_query_to_cache_field(user_input)
                    
                    if field == "detailed_summary":
                        content = f"### AI Summary for **{target_doc}**\n\n"
                        content += f"**Quick Overview:**\n{quick}\n\n"
                        content += f"**Detailed Analysis:**\n{detailed}\n\n"
                        if purpose and purpose != "N/A":
                            content += f"**Purpose:** {purpose}\n\n"
                        if audience and audience != "N/A":
                            content += f"**Intended Audience:** {audience}\n\n"
                        if difficulty and difficulty != "N/A":
                            content += f"**Difficulty Level:** {difficulty}\n\n"
                        if takeaways:
                            content += "**Key Takeaways:**\n"
                            for take in takeaways:
                                content += f"- {take}\n"
                            content += "\n"
                        if conclusion and conclusion != "N/A":
                            content += f"**Conclusion:** *{conclusion}*\n\n"
                        if topics:
                            content += "**Key Topics:** " + ", ".join([f"`{t}`" for t in topics]) + "\n"
                    else:
                        field_value = summary_data.get(field, "N/A")
                        content = f"### {field_title} for **{target_doc}**\n\n"
                        if field == "takeaways" and isinstance(field_value, list):
                            for take in field_value:
                                content += f"- {take}\n"
                        elif field == "topics" and isinstance(field_value, list):
                            content += ", ".join([f"`{t}`" for t in field_value])
                        else:
                            content += str(field_value)
                        content += "\n"
                        
                    sources = [{"name": target_doc, "page": 0}]
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": content,
                        "sources": sources,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    st.rerun()
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "No documents are indexed yet. Please upload a PDF in the sidebar first.",
                    "sources": [],
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                st.rerun()

        # Get configurable similarity threshold
        similarity_threshold = config_data.get("SIMILARITY_THRESHOLD", 1.3)

        # Retrieve relevant chunks from Chroma
        filter_dict = None
        if st.session_state.get("selected_document"):
            doc_path = get_selected_doc_path(st.session_state.selected_document)
            if doc_path:
                filter_dict = {"source": doc_path}

        assistant_response = ""
        source_type = None
        sources = []
        web_sources = []
        
        # Get chat history for context
        history_str = get_chat_history_str()

        if st.session_state.indexed_documents:
            try:
                results_with_scores = st.session_state.vectorstore.similarity_search_with_score(
                    user_input, k=4, filter=filter_dict
                )
            except Exception as e:
                logger.error(f"Error querying Chroma: {str(e)}")
                results_with_scores = []
        else:
            results_with_scores = []

        # Filter chunks by threshold
        relevant_chunks = [doc for doc, score in results_with_scores if score <= similarity_threshold]

        if relevant_chunks:
            pdf_context = "\n\n".join([doc.page_content for doc in relevant_chunks])
            pdf_ans, is_complete, can_answer, missing_info = generate_rag_answer_and_check_completeness(user_input, pdf_context, history_str)
            
            if can_answer:
                for doc in relevant_chunks:
                    source_path = doc.metadata.get("source", "Unknown")
                    source_name = os.path.basename(source_path)
                    page = doc.metadata.get("page", None)
                    source_info = {
                        "name": source_name,
                        "page": page,
                        "text": doc.page_content,
                        "id": doc.metadata.get("id") or getattr(doc, "id", "")
                    }
                    if source_info not in sources:
                        sources.append(source_info)

                if is_complete:
                    assistant_response = pdf_ans
                    source_type = "uploaded_docs"
                else:
                    # Hybrid flow
                    try:
                        web_results, provider = web_search.perform_web_search(user_input)
                        web_context = "\n\n".join([f"Source: {res['title']}\nURL: {res['link']}\nSnippet: {res['snippet']}" for res in web_results])
                        for res in web_results:
                            web_sources.append({"name": res["title"], "url": res["link"], "snippet": res.get("snippet", "")})
                    except Exception as e:
                        logger.error(f"Web search failed in hybrid flow: {str(e)}")
                        web_context = ""
                        
                    if web_context:
                        assistant_response = generate_hybrid_answer(user_input, pdf_context, web_context, history_str)
                        source_type = "hybrid"
                    else:
                        assistant_response = pdf_ans
                        source_type = "uploaded_docs"
            else:
                relevant_chunks = []

        if not relevant_chunks:
            try:
                web_results, provider = web_search.perform_web_search(user_input)
                web_context = "\n\n".join([f"Source: {res['title']}\nURL: {res['link']}\nSnippet: {res['snippet']}" for res in web_results])
                for res in web_results:
                    web_sources.append({"name": res["title"], "url": res["link"], "snippet": res.get("snippet", "")})
            except Exception as e:
                logger.error(f"Web search failed: {str(e)}")
                web_context = ""

            if web_context:
                assistant_response = generate_web_answer(user_input, web_context, history_str)
                source_type = "web"
            else:
                assistant_response = "I couldn't find reliable information in your uploaded documents or trusted web sources."
                source_type = "web"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response,
            "sources": sources,
            "web_sources": web_sources,
            "source_type": source_type,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        
        if sources:
            first_src = sources[0]
            st.session_state.selected_document = first_src["name"]
            st.session_state.active_page = first_src["page"]
            st.session_state.active_page_doc = first_src["name"]
            st.session_state.active_text = first_src.get("text", "")
            st.session_state.expand_insights = True
            
    st.rerun()

# Input box at bottom
render_input_box()