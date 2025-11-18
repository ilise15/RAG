import os
import warnings
import time
import configparser
from typing import Dict, Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

try:
    import fitz
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    import requests
    from PIL import Image
    import cv2
    import pytesseract
    from transformers import CLIPProcessor, CLIPModel
    import torch
except ImportError as e:
    print(f"Missing required dependency: {e}")

from document_processor import PDFRagQA


class ConfigManager:
    def __init__(self, config_path: str = "config.txt"):
        """Initialize configuration manager and load settings."""
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self._load_config()
    
    def _load_config(self):
        if not os.path.exists(self.config_path):
            print(f"Warning: Configuration file not found: {self.config_path}")
            print("Using default settings...")
            self._set_defaults()
            return
        
        try:
            self.config.read(self.config_path, encoding='utf-8')
            print(f"✓ Configuration loaded from: {self.config_path}")
            # Don't call _set_defaults() if config was loaded successfully
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default settings...")
            
            self.config.clear()
            self._set_defaults()
    
    def _set_defaults(self):
         if not self.config.has_section('CORE_SETTINGS'):
            self.config.add_section('CORE_SETTINGS')
        self.config.set('CORE_SETTINGS', 'embedding_model', 'all-MiniLM-L6-v2')
        self.config.set('CORE_SETTINGS', 'chunk_size', '400')
        self.config.set('CORE_SETTINGS', 'overlap_size', '50')
        self.config.set('CORE_SETTINGS', 'top_k_retrieval', '5')
        
        if not self.config.has_section('VISION_PROCESSING'):
            self.config.add_section('VISION_PROCESSING')
        self.config.set('VISION_PROCESSING', 'url', 'http://localhost:1234/v1/chat/completions')
        self.config.set('VISION_PROCESSING', 'model', 'google/gemma-3-4b')
        self.config.set('VISION_PROCESSING', 'max_workers', '3')
        self.config.set('VISION_PROCESSING', 'temperature', '0.3')
        self.config.set('VISION_PROCESSING', 'max_tokens', '300')
        
        if not self.config.has_section('LLM_SERVERS'):
            self.config.add_section('LLM_SERVERS')
        self.config.set('LLM_SERVERS', 'server1_url', 'http://localhost:1234/v1/chat/completions')
        self.config.set('LLM_SERVERS', 'server1_model', 'lm studio/models/deepseek-r1-distill-qwen-7b-q4_k_m.gguf')
        self.config.set('LLM_SERVERS', 'server1_active', 'True')
        self.config.set('LLM_SERVERS', 'server2_url', 'http://localhost:1234/v1/chat/completions')
        self.config.set('LLM_SERVERS', 'server2_model', 'liquid/lfm2-1.2b')
        self.config.set('LLM_SERVERS', 'server2_active', 'True')
        self.config.set('LLM_SERVERS', 'server3_url', 'http://localhost:1234/v1/chat/completions')
        self.config.set('LLM_SERVERS', 'server3_model', 'qwen/qwen3-14b')
        self.config.set('LLM_SERVERS', 'server3_active', 'True')
        
        if not self.config.has_section('LLM_GENERATION'):
            self.config.add_section('LLM_GENERATION')
        self.config.set('LLM_GENERATION', 'current_server', 'server1')
        self.config.set('LLM_GENERATION', 'max_tokens', '4096')
        self.config.set('LLM_GENERATION', 'temperature', '0.7')
        self.config.set('LLM_GENERATION', 'timeout', '1200')
        
        if not self.config.has_section('IMAGE_TRIAGE'):
            self.config.add_section('IMAGE_TRIAGE')
        self.config.set('IMAGE_TRIAGE', 'min_image_width', '50')
        self.config.set('IMAGE_TRIAGE', 'min_image_height', '50')
        self.config.set('IMAGE_TRIAGE', 'max_image_width', '4000')
        self.config.set('IMAGE_TRIAGE', 'max_image_height', '4000')
        self.config.set('IMAGE_TRIAGE', 'blank_image_threshold', '240')
        self.config.set('IMAGE_TRIAGE', 'edge_density_threshold', '0.01')
        self.config.set('IMAGE_TRIAGE', 'relevance_score_threshold', '0.3')
        
        if not self.config.has_section('DOCUMENT_PROCESSING'):
            self.config.add_section('DOCUMENT_PROCESSING')
        self.config.set('DOCUMENT_PROCESSING', 'min_line_length', '5')
        self.config.set('DOCUMENT_PROCESSING', 'page_limit_for_images', '3')
        
        if not self.config.has_section('PROMPTS'):
            self.config.add_section('PROMPTS')
        self.config.set('PROMPTS', 'system_prompt', 'You are an expert document analyst. Answer questions based STRICTLY on the provided document context. Analyze carefully, provide comprehensive answers, use specific details from the document, and pay attention to image descriptions.')
        self.config.set('PROMPTS', 'vision_prompt_context', 'Describe this image, focusing on content related to the question context. Include any text, diagrams, charts, or technical content.')
        self.config.set('PROMPTS', 'vision_prompt_general', 'Describe this image in detail. Focus on any text, diagrams, charts, mathematical formulas, or technical content.')
    
    def get_safe(self, section: str, option: str, fallback: str = None):
        try:
            return self.config.get(section, option)
        except:
            return fallback
    
    def get_int_safe(self, section: str, option: str, fallback: int = 0):
        try:
            return self.config.getint(section, option)
        except:
            return fallback
    
    def get_float_safe(self, section: str, option: str, fallback: float = 0.0):
        try:
            return self.config.getfloat(section, option)
        except:
            return fallback
    
    def get_bool_safe(self, section: str, option: str, fallback: bool = False):
        try:
            return self.config.getboolean(section, option)
        except:
            return fallback


class SmartPDFRagSystem:
    def __init__(self, config_path: str = "config.txt"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager
        
        self._initialize_from_config()
        
        print("Smart PDF RAG System initialized with configuration-based settings!")
    
    def _initialize_from_config(self):
        self.EMBEDDING_MODEL = self.config.get_safe('CORE_SETTINGS', 'embedding_model', 'all-MiniLM-L6-v2')
        self.CHUNK_SIZE = self.config.get_int_safe('CORE_SETTINGS', 'chunk_size', 400)
        self.OVERLAP_SIZE = self.config.get_int_safe('CORE_SETTINGS', 'overlap_size', 50)
        
        self.VISION_SERVER_CONFIG = {
            'url': self.config.get_safe('VISION_PROCESSING', 'url', 'http://localhost:1234/v1/chat/completions'),
            'model': self.config.get_safe('VISION_PROCESSING', 'model', 'google/gemma-3-4b')
        }
        self.MAX_WORKERS = self.config.get_int_safe('VISION_PROCESSING', 'max_workers', 3)
        
        self.LM_STUDIO_SERVERS = {
            'server1': {
                'url': self.config.get_safe('LLM_SERVERS', 'server1_url', 'http://localhost:1234/v1/chat/completions'),
                'model': self.config.get_safe('LLM_SERVERS', 'server1_model', 'lm studio/models/deepseek-r1-distill-qwen-7b-q4_k_m.gguf'),
                'active': self.config.get_bool_safe('LLM_SERVERS', 'server1_active', True),
                'description': 'Medium speed, medium detail'
            },
            'server2': {
                'url': self.config.get_safe('LLM_SERVERS', 'server2_url', 'http://localhost:1234/v1/chat/completions'),
                'model': self.config.get_safe('LLM_SERVERS', 'server2_model', 'liquid/lfm2-1.2b'),
                'active': self.config.get_bool_safe('LLM_SERVERS', 'server2_active', True),
                'description': 'Quick speed, lower detail'
            },
            'server3': {
                'url': self.config.get_safe('LLM_SERVERS', 'server3_url', 'http://localhost:1234/v1/chat/completions'),
                'model': self.config.get_safe('LLM_SERVERS', 'server3_model', 'qwen/qwen3-14b'),
                'active': self.config.get_bool_safe('LLM_SERVERS', 'server3_active', True),
                'description': 'Slow speed, high detail'
            }
        }
        
        self.CURRENT_SERVER = self.config.get_safe('LLM_GENERATION', 'current_server', 'server1')
        self.MAX_TOKENS = self.config.get_int_safe('LLM_GENERATION', 'max_tokens', 4096)
        self.TEMPERATURE = self.config.get_float_safe('LLM_GENERATION', 'temperature', 0.7)
        
        self.rag_system = PDFRagQA(
            embedding_model=self.EMBEDDING_MODEL,
            chunk_size=self.CHUNK_SIZE,
            overlap_size=self.OVERLAP_SIZE,
            vision_server_config=self.VISION_SERVER_CONFIG,
            lm_studio_servers=self.LM_STUDIO_SERVERS,
            current_server=self.CURRENT_SERVER,
            max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE,
            config=self.config
        )
    
    def configure_vision_server(self, url: str, model: str, max_workers: int = 3):
        self.VISION_SERVER_CONFIG['url'] = url
        self.VISION_SERVER_CONFIG['model'] = model
        self.MAX_WORKERS = max_workers
        self.rag_system.configure_vision_model(url, model, max_workers)
        print(f"Vision server reconfigured: {url} (workers: {max_workers})")
    
    def configure_lm_server(self, server_name: str, url: str, model: str = None):
        if server_name not in self.LM_STUDIO_SERVERS:
            raise ValueError(f"Server name must be 'server1', 'server2', or 'server3', got: {server_name}")
        
        self.LM_STUDIO_SERVERS[server_name]['url'] = url
        if model:
            self.LM_STUDIO_SERVERS[server_name]['model'] = model
        self.LM_STUDIO_SERVERS[server_name]['active'] = True
        
        self.rag_system.configure_lm_studio(server_name, url, model)
        print(f"LLM server {server_name} reconfigured: {url}")
    
    def switch_server(self, server_name: str):
        if server_name not in self.LM_STUDIO_SERVERS:
            raise ValueError(f"Server name must be 'server1', 'server2', or 'server3'")
        
        if not self.LM_STUDIO_SERVERS[server_name]['active']:
            raise ValueError(f"Server {server_name} is not configured/active")
        
        self.CURRENT_SERVER = server_name
        self.rag_system.switch_server(server_name)
        print(f"Switched to {server_name}")
    
    def set_generation_parameters(self, max_tokens: int = None, temperature: float = None):
        if max_tokens is not None:
            self.MAX_TOKENS = max_tokens
            self.rag_system.max_tokens = max_tokens
            
        if temperature is not None:
            self.TEMPERATURE = temperature
            self.rag_system.temperature = temperature
        
        print(f"Generation parameters updated: max_tokens={self.MAX_TOKENS}, temperature={self.TEMPERATURE}")
    
    def enable_image_processing(self, enabled: bool = True):
        self.rag_system.set_image_processing_enabled(enabled)
        status = "enabled" if enabled else "disabled"
        print(f"Image processing {status}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        return {
            'embedding_model': self.EMBEDDING_MODEL,
            'chunk_size': self.CHUNK_SIZE,
            'overlap_size': self.OVERLAP_SIZE,
            'current_server': self.CURRENT_SERVER,
            'vision_model': self.VISION_SERVER_CONFIG['model'],
            'max_workers': self.MAX_WORKERS,
            'max_tokens': self.MAX_TOKENS,
            'temperature': self.TEMPERATURE
        }
    
    def load_document(self, pdf_path: str) -> bool:
        success = self.rag_system.load_document(pdf_path)
        if success:
            info = self.rag_system.get_document_info()
            print(f"✓ Document loaded successfully:")
            print(f"  - Filename: {info['filename']}")
            print(f"  - Pages: {info['pages']}")
            print(f"  - Text segments: {info['segments']}")
            print(f"  - Relevant images: {info['relevant_images']}/{info['total_images']}")
        else:
            print("✗ Failed to load document")
        return success
    
    def ask_question(self, question: str, top_k: int = None, streaming: bool = False):
        if top_k is None:
            top_k = self.config.get_int_safe('CORE_SETTINGS', 'top_k_retrieval', 5)
            
        if streaming:
            return self.ask_question_streaming(question, top_k)
        else:
            return self.ask_question_direct(question, top_k)
    
    def ask_question_direct(self, question: str, top_k: int = 5):
        print(f"\nProcessing question: {question}")
        start_time = time.time()
        
        context_result = self.rag_system.prepare_question_context(question, top_k)
        
        if not context_result['success']:
            return {
                "answer": f"ERROR: {context_result.get('error', 'Unknown error')}",
                "sources": [],
                "confidence": 0.0,
                "server_used": None,
                "question": question
            }
        
        result = self.rag_system.ask_question_direct(question, context_result)
        
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        print(f"✓ Answer generated in {processing_time:.2f} seconds using {self.CURRENT_SERVER}")
        if context_result['images_processed'] > 0:
            print(f"✓ Processed {context_result['images_processed']} relevant images")
        
        return result
    
    def ask_question_streaming(self, question: str, top_k: int = 5):
        print(f"\nProcessing question with streaming: {question}")
        
        context_result = self.rag_system.prepare_question_context(question, top_k)
        
        if not context_result['success']:
            return {
                "error": f"ERROR: {context_result.get('error', 'Unknown error')}",
                "sources": [],
                "confidence": 0.0
            }
        
        stream_generator = self.rag_system.ask_question_streaming(question, context_result)
        
        if stream_generator is None:
            return {
                "error": "Streaming failed",
                "sources": context_result['sources'],
                "confidence": context_result['confidence']
            }
        
        return {
            "stream": stream_generator,
            "sources": context_result['sources'],
            "confidence": context_result['confidence'],
            "server_used": context_result['server_used'],
            "question": question,
            "images_processed": context_result['images_processed'],
            "pages_with_images": context_result['pages_with_images']
        }
    
    def get_document_info(self):
        return self.rag_system.get_document_info()
    
    def cleanup(self):
        self.rag_system.cleanup()
        print("System cleaned up")