# main.py
import asyncio
from src.data_loader import load_manufacturer_data, prepare_dataset
from src.process_website import process_websites
from src.image_downloader import download_batch_images
from src.utils import setup_logging, ensure_directories
from src.image_labeler import label_images
#from config.config import MLFLOW_EXPERIMENT_NAME


async def main():
    """Main pipeline execution function."""
    logger = setup_logging(__name__)
    ensure_directories()

    logger.info("Starting manufacturing image labeling pipeline")

    try:
        # Load manufacturer data
        manufacturer_data = load_manufacturer_data()

        # Process websites and get webpage contexts
        website_data = await process_websites(manufacturer_data)

        # Download images using the processed website data
        image_data = await download_batch_images(website_data)

        # Prepare dataset with webpage context
        dataset = prepare_dataset(manufacturer_data, image_data)

        # Label images
        labeled_images = label_images(dataset)

        # print(dataset)

        logger.info("Pipeline completed successfully")
        return dataset

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())