#!/usr/bin/env python3

"""
PDF RAG QA Streamlit App - Professional Interface with Conversation History
Connected to Modular System
"""

import os
import warnings
import time
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
warnings.filterwarnings("ignore")

import streamlit as st
import tempfile
from main import SmartPDFRagSystem

# Page configuration
st.set_page_config(
    page_title="PDF RAG QA System",
    layout="wide"
)

# Custom CSS for the arrow button and input styling
st.markdown("""
<style>
    /* Custom CSS for better button styling */
    .stButton > button {
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Style for arrow button */
    div[data-testid="column"] > div > div > button[kind="secondary"] {
        height: 3rem;
        width: 3rem;
        font-size: 1.5rem;
        border-radius: 50%;
        border: 1px solid #ccc;
        background-color: #f0f2f6;
        color: #262730;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 1.5rem;
    }
    
    /* Hover effect for arrow button */
    div[data-testid="column"] > div > div > button[kind="secondary"]:hover {
        background-color: #e0e2e6;
        border-color: #999;
    }
    
    /* Input area styling */
    .stTextArea textarea {
        font-size: 14px;
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'document_name' not in st.session_state:
    st.session_state.document_name = ""
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'clear_input' not in st.session_state:
    st.session_state.clear_input = False

def initialize_system():
    """Initialize the RAG system if not already done."""
    if st.session_state.rag_system is None:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = SmartPDFRagSystem()
    return st.session_state.rag_system

def load_document(uploaded_file):
    """Load PDF document into the RAG system."""
    try:
        start_time = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        rag_system = st.session_state.rag_system
        success = rag_system.load_document(tmp_file_path)
        
        os.unlink(tmp_file_path)
        
        processing_time = time.time() - start_time
        
        if success:
            st.session_state.document_loaded = True
            st.session_state.document_name = uploaded_file.name
            st.session_state.qa_history = []
            st.success(f"Document loaded in {processing_time:.2f} seconds")
            return True
        else:
            st.error(f"Failed to load document (took {processing_time:.2f}s)")
            return False
            
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return False

def format_response_time(time_seconds):
    """Format response time for display."""
    return f"{time_seconds:.2f}s"

def format_confidence(confidence):
    """Format confidence percentage for display."""
    if confidence:
        return f"{confidence:.1%}"
    return "N/A"

def display_conversation_history():
    """Display conversation history in chronological order."""
    if not st.session_state.qa_history:
        st.info("No conversation history yet. Ask a question to get started.")
        return
    
    # Display all Q&A pairs in chronological order (oldest first)
    for i, qa in enumerate(st.session_state.qa_history):
        # Question
        st.markdown(f"**Question {i+1}:**")
        st.markdown(f"{qa['question']}")
        
        # Answer
        st.markdown("**Answer:**")
        st.markdown(qa['answer'])
        
        # Metrics in a single row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Response Time", format_response_time(qa['processing_time']))
        with col2:
            st.metric("Confidence", format_confidence(qa.get('confidence')))
        with col3:
            st.metric("Images Used", qa.get('images_processed', 0))
        with col4:
            if qa.get('server_used'):
                # Convert internal server name to display name
                server_display_options = {
                    "server1": "Medium, medium detail",
                    "server2": "Fast, lower detail",
                    "server3": "Slow, high detail"
                }
                
                server_internal = qa['server_used']
                server_display = server_display_options.get(server_internal, server_internal)
                st.metric("Server", server_display.split(',')[0])
        
        # Add separator between Q&A pairs
        if i < len(st.session_state.qa_history) - 1:
            st.divider()

def main():
    """Main application with professional interface."""
    rag_system = initialize_system()
    
    # Sidebar
    with st.sidebar:
        st.header("Document Upload")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Load Document", use_container_width=True):
                with st.spinner("Loading document..."):
                    success = load_document(uploaded_file)
        
        if st.session_state.document_loaded:
            st.header("Settings")
            
            # Image processing toggle
            enable_images = st.checkbox("Enable Image Processing", value=True)
            rag_system.enable_image_processing(enable_images)
            
            # Streaming toggle
            enable_streaming = st.checkbox("Enable Live Streaming", value=True)
            
            st.subheader("Server Selection")
            
            # Create mapping between display names and internal server names
            server_display_options = {
                "Medium speed, medium detail": "server1",
                "Quick speed, lower detail": "server2", 
                "Slow speed, high detail": "server3"
            }
            
            # Get current server display name
            current_server_internal = rag_system.CURRENT_SERVER
            current_server_display = None
            for display, internal in server_display_options.items():
                if internal == current_server_internal:
                    current_server_display = display
                    break
            
            # If not found, default to first option
            if current_server_display is None:
                current_server_display = "Medium speed, medium detail"
            
            selected_server_display = st.selectbox(
                "Select LM Studio Server:",
                list(server_display_options.keys()),
                index=list(server_display_options.keys()).index(current_server_display)
            )
            
            # Get the internal server name and switch if different
            selected_server_internal = server_display_options[selected_server_display]
            if selected_server_internal != rag_system.CURRENT_SERVER:
                try:
                    rag_system.switch_server(selected_server_internal)
                    st.success(f"Switched to {selected_server_display}")
                except Exception as e:
                    st.error(f"Error switching server: {e}")
    
    # Main content area
    if st.session_state.document_loaded:
        # Document info
        st.success(f"Document loaded: {st.session_state.document_name}")
        
        doc_info = rag_system.get_document_info()
        st.info(f"Pages: {doc_info['pages']} | Segments: {doc_info['segments']} | "
                f"Images: {doc_info['total_images']} ({doc_info['relevant_images']} relevant)")
        
        # Create conversation history container
        history_container = st.container()
        with history_container:
            display_conversation_history()
        
        st.divider()
        
        # Input area at the bottom
        # Create columns for better layout - text area takes most space, button on the right
        col1, col2 = st.columns([10, 1])
        
        # Use different key each time to reset the input
        input_key = f"question_input_{len(st.session_state.qa_history)}"
        
        with col1:
            question = st.text_area(
                "Enter your question about the document:",
                key=input_key,
                height=50,
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )
        
        with col2:
            # Create a small arrow button
            ask_button = st.button("➤", help="Send question", key="send_btn")
        
        # Process question
        if ask_button and question:
            start_time = time.time()
            
            with st.spinner("Processing question and preparing context..."):
                # Use our modular system's question answering method
                if enable_streaming:
                    try:
                        result = rag_system.ask_question(question, streaming=True)
                        
                        if 'stream' in result:
                            # Handle streaming response
                            answer_placeholder = st.empty()
                            
                            with answer_placeholder.container():
                                st.markdown("**Processing your question...**")
                            
                            complete_answer = ""
                            for chunk, full_text in result['stream']:
                                complete_answer = full_text
                                # Update the placeholder with streaming content
                                with answer_placeholder.container():
                                    st.markdown("**Answer:**")
                                    st.markdown(complete_answer + " ▋")
                                time.sleep(0.02)
                            
                            # Final answer without cursor
                            with answer_placeholder.container():
                                st.markdown("**Latest Answer:**")
                                st.markdown(complete_answer)
                                
                            processing_time = time.time() - start_time
                            
                            # Store in history
                            st.session_state.qa_history.append({
                                'question': question,
                                'answer': complete_answer,
                                'confidence': result.get('confidence', 0.0),
                                'processing_time': processing_time,
                                'server_used': result.get('server_used'),
                                'images_processed': result.get('images_processed', 0)
                            })
                            
                        else:
                            # Fallback to regular if streaming has error
                            result = rag_system.ask_question(question, streaming=False)
                            
                            st.markdown("**Answer:**")
                            st.markdown(result['answer'])
                            
                            processing_time = time.time() - start_time
                            
                            # Store in history
                            st.session_state.qa_history.append({
                                'question': question,
                                'answer': result['answer'],
                                'confidence': result.get('confidence', 0.0),
                                'processing_time': processing_time,
                                'server_used': result.get('server_used'),
                                'images_processed': result.get('images_processed', 0)
                            })
                    
                    except Exception as e:
                        st.warning(f"Streaming failed, using regular mode: {e}")
                        result = rag_system.ask_question(question, streaming=False)
                        
                        st.markdown("**Answer:**")
                        st.markdown(result['answer'])
                        
                        processing_time = time.time() - start_time
                        
                        # Store in history
                        st.session_state.qa_history.append({
                            'question': question,
                            'answer': result['answer'],
                            'confidence': result.get('confidence', 0.0),
                            'processing_time': processing_time,
                            'server_used': result.get('server_used'),
                            'images_processed': result.get('images_processed', 0)
                        })
                
                else:
                    # Regular non-streaming mode
                    result = rag_system.ask_question(question, streaming=False)
                    
                    st.markdown("**Answer:**")
                    st.markdown(result['answer'])
                    
                    processing_time = time.time() - start_time
                    
                    # Store in history
                    st.session_state.qa_history.append({
                        'question': question,
                        'answer': result['answer'],
                        'confidence': result.get('confidence', 0.0),
                        'processing_time': processing_time,
                        'server_used': result.get('server_used'),
                        'images_processed': result.get('images_processed', 0)
                    })
            
            # Rerun to create new input with new key
            st.rerun()
    
    else:
        st.info("Please upload a PDF document to get started.")

if __name__ == "__main__":
    main()