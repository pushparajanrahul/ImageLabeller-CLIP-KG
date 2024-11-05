# data_loader.py
import pandas as pd
import os
from typing import Dict, List
from config.config import MANUFACTURER_DATA_PATH
from src.utils import setup_logging
from urllib.parse import urlparse

logger = setup_logging(__name__)


def format_url(url: str) -> str:
    """Ensure URL has proper format with http:// or https://"""
    if not url.startswith(('http://', 'https://')):
        return f'http://{url}'
    return url


def load_manufacturer_data(file_path: str = MANUFACTURER_DATA_PATH) -> Dict[str, Dict]:
    """Load manufacturer data from CSV."""
    logger.info(f"Loading manufacturer data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Manufacturer data file not found: {file_path}")

    df = pd.read_csv(file_path)
    manufacturer_data = {}

    for _, row in df.iterrows():
        website = row['Websites']
        if pd.notna(website):
            # Format the website URL properly
            formatted_url = format_url(website)

            # Extract products, capabilities, and industries
            products = [row[col] for col in df.columns if col.startswith('Product_') and pd.notna(row[col])]

            capabilities = [row[col] for col in df.columns if col.startswith('Process Capability_') and pd.notna(row[col])]

            industries = [row[col] for col in df.columns if col.startswith('Industry_') and pd.notna(row[col])]

            manufacturer_data[website] = {
                'Products': products,
                'Process_Capabilities': capabilities,
                'Industries': industries,
                'Website': formatted_url
            }

    logger.info(f"Loaded data for {len(manufacturer_data)} manufacturers")
    return manufacturer_data


def prepare_dataset(manufacturer_data: Dict[str, Dict], image_data: List[Dict]) -> List[Dict]:
    """Prepare dataset by matching images with manufacturer data and context."""
    dataset = []

    for img_info in image_data:
        manufacturer = img_info['manufacturer']
        if manufacturer in manufacturer_data:
            dataset.append({
                'image_path': img_info['image_path'],
                'manufacturer_data': manufacturer_data[manufacturer],
                'page_context': img_info['page_context']
            })

    logger.info(f"Prepared dataset with {len(dataset)} images")
    return dataset
