import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple
from PIL import Image
import json
from config.config import MODEL_NAME, DEVICE, IMAGE_SIZE
from src.utils import setup_logging
import re

logger = setup_logging(__name__)


class ManufacturingCLIPLabeler:
    def __init__(self):
        """Initialize the CLIP model and processor."""
        logger.info("Initializing CLIP model")
        self.device = torch.device(DEVICE)
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.max_length = 77
        self.similarity_threshold = 0.2  # Lower base threshold for better recall

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text.lower()

    def extract_visual_features(self, image_path: str) -> torch.Tensor:
        """Extract visual features with enhanced image preprocessing."""
        try:
            image = Image.open(image_path).convert('RGB')
            # Ensure proper image size
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            # Process image with CLIP processor
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                do_center_crop=True,
                do_resize=True,
                size=IMAGE_SIZE
            )
            pixel_values = inputs["pixel_values"].to(self.device)

            with torch.no_grad():
                vision_outputs = self.model.vision_model(pixel_values)
                image_features = self.model.visual_projection(vision_outputs[1])

            return image_features / image_features.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.error(f"Error in extract_visual_features for {image_path}: {str(e)}")
            raise

    def create_enhanced_prompts(self, category: str, items: List[str], context: str = "") -> List[str]:
        """Create enhanced prompts for better text-image matching."""
        base_prompts = {
            "Products": [
                "This image shows {}",
                "The product {} is visible in this image",
                "This is a manufacturing image of {}"
            ],
            "Equipment": [
                "This image shows manufacturing equipment: {}",
                "Industrial machinery and equipment: {}",
                "Manufacturing tools and equipment: {}"
            ],
            "Process": [
                "This image shows the manufacturing process of {}",
                "Manufacturing capability demonstrated: {}",
                "Industrial process shown: {}"
            ],
            "Industries": [
                "This image is related to the {} industry",
                "Industrial application in {} sector",
                "Manufacturing for {} industry"
            ]
        }

        prompts = []
        clean_context = self.clean_text(context)[:200] if context else ""

        for item in items[:3]:  # Limit to top 3 items
            for template in base_prompts.get(category, []):
                prompt = template.format(item)
                if clean_context:
                    prompt = f"{prompt}. Context: {clean_context}"
                prompts.append(prompt)

        return prompts

    def get_category_features(self, category: str, items: List[str], context: str = "") -> torch.Tensor:
        """Get features for a category with multiple prompts."""
        prompts = self.create_enhanced_prompts(category, items, context)

        all_features = []
        for prompt in prompts:
            inputs = self.processor(
                text=[prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                text_outputs = self.model.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_features = self.model.text_projection(text_outputs[1])
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_features.append(text_features)

        # Combine features by taking the max similarity
        combined_features = torch.cat(all_features, dim=0)
        return combined_features

    def calculate_similarity(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> float:
        """Calculate maximum similarity across all text features."""
        similarities = torch.cosine_similarity(
            visual_features.repeat(text_features.shape[0], 1),
            text_features,
            dim=1
        )
        return similarities.max().item()

    def label_image_with_context(self, image_path: str, manufacturer_data: Dict, page_context: str) -> Tuple[
        Dict, torch.Tensor]:
        """Label image with enhanced context processing."""
        try:
            visual_features = self.extract_visual_features(image_path)
            clean_context = self.clean_text(page_context)

            # Calculate similarities for each category
            similarities = {
                "Products": self.calculate_similarity(
                    visual_features,
                    self.get_category_features("Products", manufacturer_data['Products'], clean_context)
                ),
                "Equipment": self.calculate_similarity(
                    visual_features,
                    self.get_category_features("Equipment",
                                               ["manufacturing equipment", "industrial machinery", "tools"],
                                               clean_context)
                ),
                "Process": self.calculate_similarity(
                    visual_features,
                    self.get_category_features("Process", manufacturer_data['Process_Capabilities'], clean_context)
                ),
                "Industries": self.calculate_similarity(
                    visual_features,
                    self.get_category_features("Industries", manufacturer_data['Industries'], clean_context)
                )
            }

            # Context-based threshold adjustment
            def get_threshold(base_threshold, category, context):
                """Dynamic threshold based on context relevance."""
                context_bonus = 0.05
                category_terms = {
                    "Products": manufacturer_data['Products'],
                    "Equipment": ["equipment", "machine", "machinery", "tool"],
                    "Process": manufacturer_data['Process_Capabilities'],
                    "Industries": manufacturer_data['Industries']
                }

                terms = category_terms[category]
                if any(term.lower() in context.lower() for term in terms):
                    return base_threshold - context_bonus
                return base_threshold

            # Generate labels with adjusted thresholds
            label_output = {
                "Product Name": [
                    "1" if similarities["Products"] > get_threshold(self.similarity_threshold, "Products",
                                                                    clean_context) else "0",
                    manufacturer_data['Products'][0] if similarities["Products"] > get_threshold(
                        self.similarity_threshold, "Products", clean_context) else "None"
                ],
                "Equipment Name": [
                    "1" if similarities["Equipment"] > get_threshold(self.similarity_threshold, "Equipment",
                                                                     clean_context) else "0",
                    "Manufacturing Equipment" if similarities["Equipment"] > get_threshold(self.similarity_threshold,
                                                                                           "Equipment",
                                                                                           clean_context) else "None"
                ],
                "Process Name": [
                    "1" if similarities["Process"] > get_threshold(self.similarity_threshold, "Process",
                                                                   clean_context) else "0",
                    manufacturer_data['Process_Capabilities'][0] if similarities["Process"] > get_threshold(
                        self.similarity_threshold, "Process", clean_context) else "None"
                ],
                "Process Capabilities": [
                    "1" if similarities["Process"] > get_threshold(self.similarity_threshold, "Process",
                                                                   clean_context) else "0",
                    manufacturer_data['Process_Capabilities'][0] if similarities["Process"] > get_threshold(
                        self.similarity_threshold, "Process", clean_context) else "None"
                ],
                "Industry Name": [
                    "1" if similarities["Industries"] > get_threshold(self.similarity_threshold, "Industries",
                                                                      clean_context) else "0",
                    manufacturer_data['Industries'][0] if similarities["Industries"] > get_threshold(
                        self.similarity_threshold, "Industries", clean_context) else "None"
                ]
            }

            # Log similarities for debugging
            logger.debug(f"Similarities for {image_path}: {similarities}")

            return label_output, visual_features

        except Exception as e:
            logger.error(f"Error in label_image_with_context for {image_path}: {str(e)}")
            raise

    def label_image(self, image_path: str, manufacturer_data: Dict, page_context: str = "") -> Tuple[
        Dict, torch.Tensor]:
        """Main labeling interface."""
        try:
            return self.label_image_with_context(image_path, manufacturer_data, page_context)
        except Exception as e:
            logger.error(f"Error in label_image for {image_path}: {str(e)}")
            default_label = {
                "Product Name": ["0", "None"],
                "Equipment Name": ["0", "None"],
                "Process Name": ["0", "None"],
                "Process Capabilities": ["0", "None"],
                "Industry Name": ["0", "None"]
            }
            raise