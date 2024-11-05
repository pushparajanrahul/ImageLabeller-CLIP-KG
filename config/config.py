import os
from pathlib import Path
import torch  # Add this import

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloaded_images")
LABELED_DIR = os.path.join(DATA_DIR, "labeled_images")

# Data configuration
MANUFACTURER_DATA_PATH = os.path.join(DATA_DIR, "SUDOKN_NLP_Master.csv")
MAX_IMAGES_PER_MANUFACTURER = 10
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Model configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Download configuration
DOWNLOAD_TIMEOUT = 30
MAX_RETRIES = 3
CONCURRENT_DOWNLOADS = 5

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "manufacturing_image_labeling"