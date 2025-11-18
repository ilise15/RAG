#!/usr/bin/env python3
"""
Image Triage Module - Handles fast image filtering and triage processing
Uses simplified configuration for essential parameters.
"""

import io
import numpy as np
import cv2
import pytesseract
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any


class HybridImageTriageProcessor:
    """Fast computer vision triage processor for rapid image filtering."""
    
    def __init__(self, config=None):
        self.clip_model = None
        self.clip_processor = None
        self.enable_clip = True
        self.config = config
        
        print("Hybrid Image Triage Processor initialized")
        self.init_clip()
    
    def init_clip(self):
        """Initialize CLIP model for semantic image filtering."""
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            try:
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
            except Exception:
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model ready for semantic image filtering")
        except Exception:
            self.enable_clip = False
            print("CLIP unavailable; continuing without semantic image filtering.")
    
    def quick_image_analysis(self, image_data: bytes) -> Dict[str, Any]:
        """Rapidly analyze image to determine if it's worth processing with vision LLM."""
        try:
            pil_image = Image.open(io.BytesIO(image_data))
            width, height = pil_image.size
            
            # Use configured size thresholds if available
            min_width = min_height = 50
            max_width = max_height = 4000
            if self.config:
                min_width = self.config.get_int_safe('IMAGE_TRIAGE', 'min_image_width', 50)
                min_height = self.config.get_int_safe('IMAGE_TRIAGE', 'min_image_height', 50)
                max_width = self.config.get_int_safe('IMAGE_TRIAGE', 'max_image_width', 4000)
                max_height = self.config.get_int_safe('IMAGE_TRIAGE', 'max_image_height', 4000)
            
            if width < min_width or height < min_height:
                return {"relevant": False, "reason": "too_small", "confidence": 0.0}
            
            if width > max_width or height > max_height:
                return {"relevant": False, "reason": "too_large", "confidence": 0.0}
            
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Use configured blank image threshold if available
            blank_threshold = 240
            if self.config:
                blank_threshold = self.config.get_float_safe('IMAGE_TRIAGE', 'blank_image_threshold', 240)
            
            mean_intensity = np.mean(gray)
            if mean_intensity > blank_threshold:
                return {"relevant": False, "reason": "mostly_blank", "confidence": 0.1}
            
            # Use configured edge detection thresholds if available
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            edge_threshold = 0.01
            if self.config:
                edge_threshold = self.config.get_float_safe('IMAGE_TRIAGE', 'edge_density_threshold', 0.01)
            
            if edge_density < edge_threshold:
                return {"relevant": False, "reason": "no_structure", "confidence": 0.2}
            
            # OCR text detection
            try:
                text = pytesseract.image_to_string(pil_image, config='--psm 6')
                has_text = len(text.strip()) > 10
            except Exception:
                has_text = False
            
            # Use configured relevance threshold if available
            relevance_threshold = 0.3
            if self.config:
                relevance_threshold = self.config.get_float_safe('IMAGE_TRIAGE', 'relevance_score_threshold', 0.3)
            
            relevance_score = 0.0
            reasons = []
            
            if edge_density > 0.05:
                relevance_score += 0.4
                reasons.append("structured_content")
            
            if has_text:
                relevance_score += 0.3
                reasons.append("contains_text")
            
            if 100 < width < 2000 and 100 < height < 2000:
                relevance_score += 0.2
                reasons.append("good_size")
            
            is_relevant = relevance_score > relevance_threshold
            
            return {
                "relevant": is_relevant,
                "reason": ", ".join(reasons) if is_relevant else "low_relevance",
                "confidence": relevance_score,
                "has_text": has_text,
                "edge_density": edge_density,
                "dimensions": f"{width}x{height}"
            }
            
        except Exception as e:
            print(f"Image triage fallback due to error: {e}")
            return {"relevant": True, "reason": "analysis_error", "confidence": 0.5}