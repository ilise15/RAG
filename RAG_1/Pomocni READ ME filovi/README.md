# Smart PDF RAG QA System - Configuration-Based Version

A comprehensive PDF question-answering system with advanced image processing, multi-server LLM support, configuration management, and professional Streamlit frontend.

## 🏗️ Project Architecture

```
smart-pdf-rag-system/
├── main.py                  # Main integration with centralized config management
├── document_processor.py    # PDF processing and RAG system core
├── vision_processor.py      # Image extraction and AI description
├── image_triage.py         # Image filtering and relevance analysis
├── app.py                  # Streamlit frontend application
├── config.txt              # Configuration file (all settings)
├── requirements.txt        # Python dependencies
└── README.md              # This documentation
```

## ✨ Key Features

### **Configuration-Driven Architecture**
- **Centralized Configuration**: All settings managed via `config.txt` file
- **Safe Configuration Loading**: Automatic fallbacks to defaults if config missing
- **Runtime Configuration**: Modify settings without code changes
- **Configurable Prompts**: LLM prompt templates fully customizable

### **Advanced Document Processing**
- **Modular Design**: Clean separation of concerns across focused modules
- **Hybrid Image Triage**: Intelligent filtering processes only relevant images
- **Multi-Server LLM Support**: Three configurable servers with different performance profiles
- **Streaming Responses**: Real-time answer generation with live updates
- **Vector Search**: FAISS-based semantic search for accurate context retrieval

### **Professional User Interface**
- **Streamlit Frontend**: Clean, responsive interface with conversation history
- **Server Selection**: Easy switching between LLM servers during runtime
- **Processing Controls**: Toggle image processing and streaming modes
- **Performance Metrics**: Response time, confidence scores, and processing statistics

### **Image Processing Pipeline**
- **CLIP Integration**: Semantic image understanding and relevance scoring
- **OCR Text Extraction**: Automatic text detection in images and diagrams
- **Computer Vision**: Edge detection and structural analysis for content filtering
- **Concurrent Processing**: Multi-threaded image processing for performance

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
git clone <repository-url> # or download files manually

# Install Python dependencies
pip install -r requirements.txt
```

### 2. System Dependencies

**Tesseract OCR** (Required for image text extraction):
- **Windows**: Download from [Tesseract Wiki](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

**LM Studio** (Required for LLM functionality):
- Download from [LM Studio](https://lmstudio.ai/)
- Load your preferred models (DeepSeek, Qwen, Llama, etc.)
- Start server on `http://localhost:1234`

### 3. Configuration Setup

The system uses `config.txt` for all configuration. Default settings work out-of-the-box, but you can customize:

```ini
[CORE_SETTINGS]
embedding_model = all-MiniLM-L6-v2
chunk_size = 400
overlap_size = 50
top_k_retrieval = 5

[VISION_PROCESSING]
url = http://localhost:1234/v1/chat/completions
model = google/gemma-3-4b
max_workers = 3

[LLM_SERVERS]
server1_url = http://localhost:1234/v1/chat/completions
server1_model = your-model-name-here
server1_active = True

[PROMPTS]
system_prompt = You are an expert document analyst. Answer questions based STRICTLY on the provided document context.
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 📖 Usage Guide

### **Document Upload**
1. Click "Choose a PDF file" in the sidebar
2. Select your PDF document
3. Click "Load Document" - the system will:
   - Extract and process text content
   - Analyze and triage all images
   - Build vector embeddings for search
   - Display processing summary

### **Asking Questions**
1. Enable/disable **Image Processing** for your use case
2. Toggle **Live Streaming** for real-time response generation  
3. Select **LLM Server** based on your performance needs:
   - **Server 1**: Medium speed, medium detail
   - **Server 2**: Quick speed, lower detail  
   - **Server 3**: Slow speed, high detail
4. Type your question and click the arrow button
5. View the response with performance metrics

### **Conversation History**
- All Q&A pairs are preserved in chronological order
- Each interaction shows:
  - **Response Time**: Processing duration
  - **Confidence Score**: Similarity-based confidence
  - **Images Used**: Number of images processed
  - **Server Used**: Which LLM server generated the response

## ⚙️ Configuration Reference

### **Core Settings**
- `embedding_model`: Sentence transformer model for embeddings
- `chunk_size`: Text segment size in words
- `overlap_size`: Word overlap between segments
- `top_k_retrieval`: Number of segments to retrieve

### **Vision Processing**
- `url`: Vision LLM server endpoint
- `model`: Vision model name
- `max_workers`: Concurrent image processing threads
- `temperature`: Vision model creativity (0.0-1.0)
- `max_tokens`: Maximum tokens for image descriptions

### **LLM Servers**
Configure up to 3 servers with different models:
- `server1_url/model/active`: Primary server configuration
- `server2_url/model/active`: Fast processing server
- `server3_url/model/active`: High-quality server

### **Image Triage**
Fine-tune image filtering parameters:
- `min/max_image_width/height`: Size filtering bounds
- `blank_image_threshold`: Brightness threshold for blank detection
- `edge_density_threshold`: Structural content detection
- `relevance_score_threshold`: Minimum relevance score

### **Prompts** (Most Important)
Customize LLM behavior with configurable prompts:
- `system_prompt`: Instructions for document analysis
- `vision_prompt_context`: Context-aware image description
- `vision_prompt_general`: General image description

## 🔧 Advanced Configuration

### **Performance Optimization**
```ini
[VISION_PROCESSING]
max_workers = 6              # Increase for more CPU cores

[IMAGE_TRIAGE]
relevance_score_threshold = 0.5  # Higher = more selective

[DOCUMENT_PROCESSING]
page_limit_for_images = 5    # More pages = better context
```

### **Model Customization**
```ini
[LLM_SERVERS]
server1_model = deepseek-coder-33b-instruct
server2_model = qwen2.5-7b-instruct  
server3_model = llama-3.1-70b-instruct

[LLM_GENERATION]
max_tokens = 8192            # Longer responses
temperature = 0.3            # More focused answers
```

### **Prompt Engineering**
```ini
[PROMPTS]
system_prompt = You are a specialized technical document analyst with expertise in [YOUR DOMAIN]. Focus on technical accuracy and cite specific sections when answering questions about the provided document context.
```

## 🛠️ Programmatic Usage

```python
from main import SmartPDFRagSystem

# Initialize with custom config
system = SmartPDFRagSystem("my_config.txt")

# Load document
success = system.load_document("document.pdf")

if success:
    # Ask questions
    result = system.ask_question("What are the key findings?")
    print(result['answer'])
    
    # Use streaming
    result = system.ask_question("Explain the methodology", streaming=True)
    if 'stream' in result:
        for chunk, complete in result['stream']:
            print(chunk, end='', flush=True)
    
    # Runtime reconfiguration
    system.switch_server('server2')  # Switch to fast server
    system.enable_image_processing(False)  # Disable images
```

## 🐛 Troubleshooting

### **Common Issues**

**ConfigParser Error**: Config file format issues
```bash
# Solution: Check config.txt format, ensure no special characters in prompts
# System automatically falls back to defaults if config is invalid
```

**FAISS Installation**: Linux compatibility issues
```bash
pip install faiss-cpu --no-cache
# Or for GPU: pip install faiss-gpu --no-cache
```

**Tesseract Not Found**: OCR functionality unavailable
```bash
# Windows: Add Tesseract to PATH environment variable
# Linux/macOS: Verify installation with `tesseract --version`
```

**LM Studio Connection**: Server communication errors
```bash
# 1. Verify LM Studio is running on localhost:1234
# 2. Check that models are loaded
# 3. Test server endpoint manually
```

### **Performance Issues**

**Slow Image Processing**:
- Reduce `max_workers` for memory-constrained systems
- Increase `relevance_score_threshold` to process fewer images
- Set `page_limit_for_images` to limit image processing scope

**Memory Usage**:
- Use smaller embedding models
- Reduce `chunk_size` for large documents
- Switch to `faiss-cpu` if using GPU version

**Response Quality**:
- Experiment with different `temperature` settings
- Adjust `system_prompt` for your specific domain
- Try different servers for quality vs. speed tradeoffs

## 🏁 System Requirements

### **Minimum Requirements**
- **Python**: 3.8+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB for models and dependencies
- **Network**: Internet connection for model downloads

### **Recommended Requirements**
- **Python**: 3.10+
- **RAM**: 16GB+ for large documents and concurrent processing
- **GPU**: CUDA-compatible GPU for faster processing (optional)
- **CPU**: Multi-core processor for concurrent image processing

## 🤝 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Verify all dependencies are correctly installed
3. Test with a simple PDF document first
4. Review configuration settings in `config.txt`

The system is designed to be robust with automatic fallbacks and comprehensive error handling, making it suitable for both development and production use.