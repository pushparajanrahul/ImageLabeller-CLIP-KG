# image_labeler.py
from typing import List, Dict
import os
import json
from tqdm import tqdm
from config.config import LABELED_DIR
from src.model import ManufacturingCLIPLabeler
from src.utils import setup_logging

logger = setup_logging(__name__)


def label_images(dataset: List[Dict]) -> List[Dict]:
    """Label all images in the dataset using both image and context information."""
    logger.info("Starting image labeling process")
    os.makedirs(LABELED_DIR, exist_ok=True)

    labeler = ManufacturingCLIPLabeler()
    labeled_images = []

    for item in tqdm(dataset, desc="Labeling images"):
        image_path = item['image_path']
        manufacturer_data = item['manufacturer_data']
        page_context = item.get('page_context', '')  # Get context if available

        try:
            # Label image using both visual and contextual information
            label_output, _ = labeler.label_image(
                image_path=image_path,
                manufacturer_data=manufacturer_data,
                page_context=page_context
            )

            # Save label to file
            label_path = os.path.join(
                LABELED_DIR,
                f"{os.path.basename(image_path)}.json"
            )
            with open(label_path, 'w') as f:
                json.dump(label_output, f, indent=2)

            labeled_images.append({
                'image_path': image_path,
                'label_path': label_path,
                'label': label_output
            })

        except Exception as e:
            logger.error(f"Error labeling image {image_path}: {str(e)}")

    logger.info(f"Labeled {len(labeled_images)} images")
    return labeled_images