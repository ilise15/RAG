#!/usr/bin/env python3
"""
Document Processing Module - Handles PDF text extraction, cleaning and RAG processing
Uses simplified configuration for essential parameters.
"""

import os
import re
import json
import time
import tempfile
import shutil
from typing import List, Dict, Any, Tuple, Optional, Generator
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
from vision_processor import SmartVisionProcessor


class PDFRagQA:
    def __init__(self, embedding_model: str, chunk_size: int, overlap_size: int, 
                 vision_server_config: Dict[str, str], lm_studio_servers: Dict[str, Dict],
                 current_server: str, max_tokens: int, temperature: float, config=None):
        
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.embedding_model = None
        self.segments = []
        self.embeddings = None
        self.faiss_index = None
        self.document_metadata = {}
        self.current_pdf_path = None
        self.persistent_pdf_path = None
        
        # Store configuration manager
        self.config = config
        
        self.vision_processor = SmartVisionProcessor(vision_server_config, config=config)
        
        self.lm_studio_servers = lm_studio_servers
        self.current_server = current_server
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print("Initializing Smart PDF RAG QA System with Upload-Time Hybrid Triage, Multiple Servers and Streaming Support...")
        self.load_models()
    
    def set_image_processing_enabled(self, enabled: bool):
        self.vision_processor.set_image_processing_enabled(enabled)
    
    def configure_vision_model(self, url: str, model: str, max_workers: int = 3):
        self.vision_processor.configure_vision_server(url, model, max_workers)
    
    def load_models(self):
        try:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("Models loaded successfully!")
        except Exception as e:
            print(f"ERROR: Error loading models: {e}")
    
    def configure_lm_studio(self, server_name: str, url: str, model: str = None):
        if server_name not in self.lm_studio_servers:
            raise ValueError(f"Server name must be 'server1', 'server2', or 'server3', got: {server_name}")
        
        self.lm_studio_servers[server_name]['url'] = url
        if model:
            self.lm_studio_servers[server_name]['model'] = model
        self.lm_studio_servers[server_name]['active'] = True
        
        print(f"Configured {server_name}: {url}")
    
    def switch_server(self, server_name: str):
        if server_name not in self.lm_studio_servers:
            raise ValueError(f"Server name must be 'server1', 'server2', or 'server3', got: {server_name}")
        
        if not self.lm_studio_servers[server_name]['active']:
            raise ValueError(f"Server {server_name} is not configured/active")
        
        self.current_server = server_name
        print(f"Switched to {server_name}")
    
    def _create_persistent_copy(self, original_path: str) -> str:
        try:
            temp_dir = tempfile.gettempdir()
            import uuid
            file_id = str(uuid.uuid4())[:8]
            persistent_path = os.path.join(temp_dir, f"pdf_rag_{file_id}.pdf")
            shutil.copy2(original_path, persistent_path)
            return persistent_path
        except Exception:
            return original_path
    
    def extract_pdf_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        text_data = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                text = self.clean_text(text)
                if text.strip():
                    text_data.append({"page_number": page_num + 1, "text": text, "word_count": len(text.split())})
            doc.close()
            print(f"Extracted {len(text_data)} pages")
        except Exception as e:
            print(f"ERROR: Error extracting PDF text: {e}")
        return text_data
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\t+', ' ', text)
        lines = text.split('\n')
        cleaned_lines = []
        
        # Use configured minimum line length if available
        min_length = 5
        if self.config:
            min_length = self.config.get_int_safe('DOCUMENT_PROCESSING', 'min_line_length', 5)
        
        for line in lines:
            line = line.strip()
            if len(line) > min_length and not re.match(r'^\d+$', line):
                cleaned_lines.append(line)
        return ' '.join(cleaned_lines)
    
    def create_segments(self, text_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        segments = []
        segment_id = 0
        for page_data in text_data:
            page_num = page_data["page_number"]
            text = page_data["text"]
            words = text.split()
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                segment_text = " ".join(words[start:end])
                segments.append({
                    "id": segment_id,
                    "text": segment_text,
                    "page_number": page_num,
                    "start_word": start,
                    "end_word": end,
                    "word_count": len(segment_text.split())
                })
                segment_id += 1
                if end >= len(words):
                    break
                start += self.chunk_size - self.overlap_size
        return segments
    
    def build_vector_index(self):
        if not self.segments:
            print("ERROR: No segments to index")
            return
        
        texts = [segment["text"] for segment in self.segments]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)
    
    def load_document(self, pdf_path: str) -> bool:
        if not os.path.exists(pdf_path):
            return False
        
        try:
            if "tmp" in pdf_path.lower() or "temp" in pdf_path.lower():
                self.persistent_pdf_path = self._create_persistent_copy(pdf_path)
                self.current_pdf_path = self.persistent_pdf_path
            else:
                self.current_pdf_path = pdf_path
                self.persistent_pdf_path = pdf_path
            
            text_data = self.extract_pdf_text(self.current_pdf_path)
            if not text_data:
                return False
            
            image_metadata = self.vision_processor.scan_and_triage_pdf_images(self.current_pdf_path)
            
            self.segments = self.create_segments(text_data)
            self.build_vector_index()
            
            total_images = sum(meta['count'] for meta in image_metadata.values())
            relevant_images = sum(len([r for r in meta.get('triage_results', []) if r['triage_result']['relevant']]) for meta in image_metadata.values())
            
            print(f"Pre-triaged {relevant_images} relevant images out of {total_images} total images")
            
            self.document_metadata = {
                "path": self.current_pdf_path,
                "filename": os.path.basename(pdf_path),
                "pages": len(text_data),
                "segments": len(self.segments),
                "pages_with_images": len(image_metadata),
                "total_images": total_images,
                "relevant_images": relevant_images
            }
            
            return True
            
        except Exception as e:
            print(f"ERROR: Error loading document: {e}")
            return False
    
    def search_relevant_segments(self, question: str, top_k: int) -> List[Tuple[Dict, float]]:
        question_embedding = self.embedding_model.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(question_embedding)
        
        distances, indices = self.faiss_index.search(question_embedding, top_k)
        
        relevant_segments = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.segments):
                segment = self.segments[idx]
                cosine_similarity = 1.0 - (distance ** 2) / 2.0
                cosine_similarity = max(0.0, min(1.0, cosine_similarity))
                similarity_percent = cosine_similarity * 100.0
                relevant_segments.append((segment, similarity_percent))
        
        return relevant_segments
    
    def prepare_question_context(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Prepare context for question using configuration-based settings."""
        if not self.segments or self.faiss_index is None:
            return {"success": False, "error": "No document loaded"}
        
        try:
            # Find relevant text segments
            relevant_segments = self.search_relevant_segments(question, top_k)
            
            if not relevant_segments:
                return {"success": False, "error": "No relevant content found"}
            
            # Get pages from relevant segments
            relevant_pages = list(set(segment['page_number'] for segment, _ in relevant_segments))
            print(f"Found relevant content on pages: {relevant_pages}")
            
            # Check if any relevant pages have images and if image processing is enabled
            pages_with_images = self.vision_processor.get_pages_with_images()
            image_pages_to_process = [p for p in relevant_pages if p in pages_with_images]
            
            # Use configured page limit for images
            page_limit = 3
            if self.config:
                page_limit = self.config.get_int_safe('DOCUMENT_PROCESSING', 'page_limit_for_images', 3)
            
            # Process only pre-triaged relevant images for relevant pages
            image_descriptions = []
            if image_pages_to_process and self.current_pdf_path and self.vision_processor.image_processing_enabled:
                print(f"Processing pre-triaged images on {len(image_pages_to_process)} relevant pages...")
                image_descriptions = self.vision_processor.process_images_for_pages(
                    self.current_pdf_path,
                    image_pages_to_process[:page_limit],
                    question
                )
            else:
                print("Image processing disabled - using text-only mode")
            
            # Prepare context with text and relevant images
            context_parts = []
            sources = []
            
            for segment, similarity in relevant_segments:
                context_parts.append(f"Page {segment['page_number']}: {segment['text']}")
                sources.append({
                    "page": segment["page_number"],
                    "segment_id": segment["id"],
                    "similarity": similarity,
                    "text_preview": segment['text'][:100] + "..." if len(segment['text']) > 100 else segment['text']
                })
            
            # Add image descriptions to context
            for img_desc in image_descriptions:
                context_parts.append(f"Page {img_desc['page_number']} - [Image {img_desc['image_index']}: {img_desc['description']}]")
            
            context = "\n\n".join(context_parts)
            avg_confidence = sum(sim for _, sim in relevant_segments) / len(relevant_segments) / 100.0
            
            return {
                "success": True,
                "context": context,
                "sources": sources,
                "confidence": float(avg_confidence),
                "server_used": self.current_server,
                "question": question,
                "images_processed": len(image_descriptions),
                "pages_with_images": len(image_pages_to_process)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error preparing context: {e}"}
    
    def _get_system_prompt(self) -> str:
        """Get system prompt from config or use default."""
        if self.config:
            return self.config.get_safe('PROMPTS', 'system_prompt', 
                'You are an expert document analyst. Answer questions based STRICTLY on the provided document context.')
        else:
            return """You are an expert document analyst. Answer questions based STRICTLY on the provided document context.

INSTRUCTIONS:
1. Analyze the context carefully
2. Provide comprehensive, well-structured answers
3. If the context doesn't contain the answer, say "The document doesn't contain information about this."
4. Use specific details and evidence from the document
5. Be thorough and detailed in your explanations
6. Quote relevant parts when helpful
7. Pay attention to image descriptions marked with [Image X: ...] as they contain important visual information"""
    
    def _get_user_prompt(self, question: str, context: str) -> str:
        """Get user prompt from config or use default."""
        return f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION: {question}\n\nPlease provide a comprehensive answer based on the document context above."
    
    def generate_with_lm_studio_streaming(self, question: str, context: str) -> Optional[Generator[Tuple[str, str], None, None]]:
        """Generate answer using current LM Studio server with streaming."""
        server_config = self.lm_studio_servers[self.current_server]
        
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(question, context)
        
        payload = {
            "model": server_config['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        try:
            timeout = 1200
            if self.config:
                timeout = self.config.get_int_safe('LLM_GENERATION', 'timeout', 1200)
            
            response = requests.post(
                server_config['url'],
                json=payload,
                timeout=timeout,
                stream=True
            )
            response.raise_for_status()
            
            def stream_response():
                complete_response = ""
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and data['choices']:
                                    chunk = data['choices'][0].get('delta', {}).get('content', '')
                                    if chunk:
                                        complete_response += chunk
                                        yield chunk, complete_response
                                    time.sleep(0.02)
                            except json.JSONDecodeError:
                                continue
            
            return stream_response()
            
        except Exception as e:
            print(f"Streaming failed: {e}")
            return None
    
    def generate_with_lm_studio(self, question: str, context: str) -> str:
        """Generate answer using current LM Studio server (non-streaming)."""
        server_config = self.lm_studio_servers[self.current_server]
        
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(question, context)
        
        payload = {
            "model": server_config['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            timeout = 1200
            if self.config:
                timeout = self.config.get_int_safe('LLM_GENERATION', 'timeout', 1200)
            
            response = requests.post(server_config['url'], json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error communicating with {self.current_server}: {e}"
    
    def ask_question_streaming(self, question: str, context_result: Dict[str, Any]) -> Optional[Generator[Tuple[str, str], None, None]]:
        """Ask a question with streaming response using pre-prepared context."""
        if not context_result['success']:
            return None
        
        return self.generate_with_lm_studio_streaming(question, context_result['context'])
    
    def ask_question_direct(self, question: str, context_result: Dict[str, Any]) -> Dict[str, Any]:
        """Ask a question with direct response using pre-prepared context."""
        if not context_result['success']:
            return {"answer": f"ERROR: {context_result.get('error', 'Unknown error')}", "sources": [], "confidence": 0.0}
        
        answer = self.generate_with_lm_studio(question, context_result['context'])
        
        return {
            "answer": answer,
            "sources": context_result['sources'],
            "confidence": context_result['confidence'],
            "server_used": context_result['server_used'],
            "images_processed": context_result['images_processed']
        }
    
    def ask_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Original non-streaming ask_question method for backwards compatibility."""
        context_result = self.prepare_question_context(question, top_k)
        
        if not context_result['success']:
            return {"answer": f"ERROR: {context_result.get('error', 'Unknown error')}", "sources": [], "confidence": 0.0, "server_used": None, "question": question}
        
        start_time = time.time()
        answer = self.generate_with_lm_studio(question, context_result['context'])
        processing_time = time.time() - start_time
        
        print(f"✓ Answer generated in {processing_time:.2f} seconds using {self.current_server}")
        if context_result['images_processed'] > 0:
            print(f"✓ Processed {context_result['images_processed']} relevant images")
        
        return {
            "answer": answer,
            "sources": context_result['sources'],
            "confidence": context_result['confidence'],
            "server_used": context_result['server_used'],
            "question": question,
            "images_processed": context_result['images_processed'],
            "pages_with_images": context_result['pages_with_images'],
            "image_processing_enabled": self.vision_processor.image_processing_enabled,
            "processing_time": processing_time
        }
    
    def get_document_info(self) -> Dict[str, Any]:
        info = self.document_metadata.copy()
        info["lm_studio_config"] = {"current_server": self.current_server, "servers": self.lm_studio_servers, "max_tokens": self.max_tokens, "temperature": self.temperature}
        info["image_processing_enabled"] = self.vision_processor.image_processing_enabled
        return info
    
    def cleanup(self):
        try:
            if self.persistent_pdf_path and self.persistent_pdf_path != self.current_pdf_path:
                if os.path.exists(self.persistent_pdf_path):
                    os.remove(self.persistent_pdf_path)
        except Exception:
            pass