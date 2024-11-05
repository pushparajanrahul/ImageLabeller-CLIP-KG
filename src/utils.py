# utils.py
import logging
import os
from datetime import datetime


def setup_logging(name: str = __name__) -> logging.Logger:
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        f'logs/manufacturing_labeler_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def ensure_directories():
    """Ensure all necessary directories exist."""
    directories = ['data', 'data/downloaded_images', 'data/labeled_images', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def generate_image_name(manufacturer_index: int, image_count: int) -> str:
    """Generate structured image name following SDK[AAA]0[ANN] pattern."""

    # Convert manufacturer index to three letters (AAA-ZZZ)
    def index_to_letters(idx):
        letters = ''
        for _ in range(3):
            letters = chr(65 + (idx % 26)) + letters  # 65 is ASCII for 'A'
            idx //= 26
        return letters

    # Convert image count to section and number
    section = chr(65 + ((image_count - 1) // 99))  # A-Z based on count
    number = ((image_count - 1) % 99) + 1  # 1-99 for each section

    return f"SDK{index_to_letters(manufacturer_index)}0{section}{number:02d}"