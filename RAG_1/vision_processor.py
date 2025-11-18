#!/usr/bin/env python3
"""
Vision Processing Module - Handles image extraction, processing and description generation
Now prints enable/disable message only on actual state change.
"""

import os
import base64
import fitz
import requests
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from image_triage import HybridImageTriageProcessor


class SmartVisionProcessor:
    """Smart image processor with hybrid triage and on-demand processing."""

    def __init__(self, vision_server_config: Dict[str, str], max_workers: int = 3, config=None):
        self.vision_server_config = vision_server_config
        self.max_workers = max_workers
        self.image_cache = {}
        self.image_metadata = {}
        self.config = config
        self.triage_processor = HybridImageTriageProcessor(config=config)
        self.image_processing_enabled = True

    def set_image_processing_enabled(self, enabled: bool):
        """Enable or disable image processing, printing message only on state change."""
        if enabled != self.image_processing_enabled:
            self.image_processing_enabled = enabled
            print(f"Image processing {'enabled' if enabled else 'disabled'}")

    def configure_vision_server(self, url: str, model: str, max_workers: int = 3):
        self.vision_server_config['url'] = url
        self.vision_server_config['model'] = model
        self.max_workers = max_workers
        print(f"Vision server configured: {url} (workers: {max_workers})")

    def scan_and_triage_pdf_images(self, pdf_path: str):
        self.image_metadata = {}

        if not os.path.exists(pdf_path):
            return {}

        print("Running hybrid triage on all images during upload...")

        doc = fitz.open(pdf_path)
        total_images = 0
        triaged_images = 0

        for page_num, page in enumerate(doc):
            image_list = page.get_images()

            if image_list:
                page_triage_results = []

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                            triage_result = self.triage_processor.quick_image_analysis(img_data)

                            page_triage_results.append({
                                'image_index': img_index,
                                'triage_result': triage_result,
                                'image_info': img[1:5]
                            })

                            if triage_result['relevant']:
                                triaged_images += 1
                                print(f" ✓ Page {page_num + 1}, Image {img_index}: {triage_result['reason']} "
                                      f"(score: {triage_result['confidence']:.2f})")
                            else:
                                print(f" ✗ Page {page_num + 1}, Image {img_index}: {triage_result['reason']}")
                        pix = None

                    except Exception as e:
                        print(f"⚠ Could not triage image {img_index} on page {page_num + 1}: {e}")
                        page_triage_results.append({
                            'image_index': img_index,
                            'triage_result': {"relevant": True, "reason": "triage_error", "confidence": 0.5},
                            'image_info': img[1:5]
                        })
                        triaged_images += 1

                self.image_metadata[page_num + 1] = {
                    'count': len(image_list),
                    'images': image_list,
                    'triage_results': page_triage_results
                }
                total_images += len(image_list)

        doc.close()
        print(f"Triage complete: {triaged_images} relevant images out of {total_images} total")
        return self.image_metadata

    def get_pages_with_images(self) -> List[int]:
        return list(self.image_metadata.keys())

    def get_relevant_images_for_pages(self, page_numbers: List[int]) -> List[Dict]:
        relevant_images = []

        for page_num in page_numbers:
            if page_num in self.image_metadata:
                page_triage_results = self.image_metadata[page_num].get('triage_results', [])

                for result in page_triage_results:
                    if result['triage_result']['relevant']:
                        relevant_images.append({
                            'page_number': page_num,
                            'image_index': result['image_index'],
                            'bbox': result['image_info'],
                            'triage_score': result['triage_result']['confidence']
                        })

        print(f"Found {len(relevant_images)} pre-triaged relevant images for processing")
        return relevant_images

    def process_images_for_pages(self, pdf_path: str, page_numbers: List[int], question_context: str = "") -> List[Dict[str, Any]]:
        if not self.image_processing_enabled:
            print("Image processing disabled - skipping all images")
            return []

        if not os.path.exists(pdf_path):
            print(f"✗ PDF file not found for image processing: {pdf_path}")
            return []

        relevant_image_refs = self.get_relevant_images_for_pages(page_numbers)

        if not relevant_image_refs:
            print("No pre-triaged relevant images found for processing")
            return []

        images_to_process = []
        doc = fitz.open(pdf_path)

        for img_ref in relevant_image_refs:
            try:
                page = doc[img_ref['page_number'] - 1]
                image_list = page.get_images()

                if img_ref['image_index'] < len(image_list):
                    img = image_list[img_ref['image_index']]
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        cache_key = f"{img_ref['page_number']}_{img_ref['image_index']}"

                        if cache_key not in self.image_cache:
                            images_to_process.append({
                                'page_number': img_ref['page_number'],
                                'image_index': img_ref['image_index'],
                                'image_data': img_data,
                                'bbox': img_ref['bbox'],
                                'cache_key': cache_key
                            })
                    pix = None

            except Exception as e:
                print(f"⚠ Could not extract image {img_ref['image_index']} on page {img_ref['page_number']}: {e}")
                continue

        doc.close()

        if not images_to_process:
            return []

        print(f"Processing {len(images_to_process)} pre-triaged relevant images...")

        max_workers = self.max_workers
        if self.config:
            max_workers = self.config.get_int_safe('VISION_PROCESSING', 'max_workers', self.max_workers)

        processed_images = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(
                    self.describe_image_with_lm_studio,
                    img_info['image_data'],
                    img_info,
                    question_context
                ): img_info for img_info in images_to_process
            }

            for future in as_completed(future_to_image):
                try:
                    result = future.result()
                    processed_images.append(result)
                    cache_key = future_to_image[future]['cache_key']
                    self.image_cache[cache_key] = result
                except Exception as exc:
                    print(f"✗ Error processing image: {exc}")

        print(f"Final result: {len(processed_images)} images processed successfully")
        return processed_images

    def describe_image_with_lm_studio(self, image_data: bytes, image_info: Dict[str, Any], question_context: str = "") -> Dict[str, Any]:
        try:
            print(f"  Processing image {image_info['image_index']} on page {image_info['page_number']}...")

            base64_image = base64.b64encode(image_data).decode('utf-8')

            if self.config and question_context:
                prompt = self.config.get_safe('PROMPTS', 'vision_prompt_context',
                                              'Describe this image, focusing on content related to the question context. Include any text, diagrams, charts, or technical content.')
            elif self.config:
                prompt = self.config.get_safe('PROMPTS', 'vision_prompt_general',
                                              'Describe this image in detail. Focus on any text, diagrams, charts, mathematical formulas, or technical content.')
            else:
                if question_context:
                    prompt = f"Describe this image, focusing on content related to: {question_context}. Include any text, diagrams, charts, or technical content."
                else:
                    prompt = "Describe this image in detail. Focus on any text, diagrams, charts, mathematical formulas, or technical content."

            temperature = self.config.get_float_safe('VISION_PROCESSING', 'temperature', 0.3) if self.config else 0.3
            max_tokens = self.config.get_int_safe('VISION_PROCESSING', 'max_tokens', 300) if self.config else 300

            payload = {
                "model": self.vision_server_config['model'],
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            response = requests.post(
                self.vision_server_config['url'],
                json=payload,
                timeout=1200
            )
            response.raise_for_status()

            result = response.json()
            description = result["choices"][0]["message"]["content"].strip()

            print(f"  ✓ Completed image {image_info['image_index']} on page {image_info['page_number']}")

            return {
                'page_number': image_info['page_number'],
                'image_index': image_info['image_index'],
                'description': description,
                'bbox': image_info['bbox']
            }

        except Exception as e:
            error_msg = f"Error processing image: {e}"
            print(f"  ✗ Failed image {image_info['image_index']}: {error_msg}")

            return {
                'page_number': image_info['page_number'],
                'image_index': image_info['image_index'],
                'description': error_msg,
                'bbox': image_info['bbox']
            }
