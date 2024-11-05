# image_downloader.py
import aiohttp
import asyncio
import aiofiles
import os
from typing import Dict, List
from config.config import DOWNLOAD_DIR
from src.utils import setup_logging, generate_image_name

logger = setup_logging(__name__)


async def download_single_image(session: aiohttp.ClientSession,
                                image_data: Dict,
                                save_path: str,
                                image_counter: int) -> Dict:
    """Download a single image and return its information."""
    try:
        async with session.get(image_data['url'], ssl=False) as response:
            if response.status == 200:
                content = await response.read()

                os.makedirs(save_path, exist_ok=True)
                image_name = generate_image_name(
                    image_data['manufacturer_index'],
                    image_counter
                )

                # Get file extension from URL or default to .jpg
                ext = os.path.splitext(image_data['url'])[1].lower() or '.jpg'
                if ext not in ['.jpg', '.jpeg', '.png', '.gif']:
                    ext = '.jpg'

                file_path = os.path.join(save_path, f"{image_name}{ext}")

                async with aiofiles.open(file_path, mode='wb') as f:
                    await f.write(content)

                return {
                    'manufacturer': image_data['manufacturer'],
                    'image_path': file_path,
                    'alt_text': image_data.get('alt_text', ''),
                    'source_type': image_data.get('source', ''),
                    'page_context': image_data.get('page_context', '')
                }
    except Exception as e:
        logger.error(f"Error downloading image {image_data['url']}: {str(e)}")
    return None


async def download_batch_images(website_data: List[Dict]) -> List[Dict]:
    """Download images from processed website data."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        image_counter = {}  # Track image count per manufacturer

        for site_data in website_data:
            manufacturer = site_data['manufacturer']
            save_path = os.path.join(DOWNLOAD_DIR, manufacturer)

            # Initialize counter for this manufacturer if not exists
            if manufacturer not in image_counter:
                image_counter[manufacturer] = 1

            for img in site_data['images']:
                # Skip if we've reached maximum images for this manufacturer
                if image_counter[manufacturer] > 26 * 99:  # Z99 limit
                    continue

                img_data = {
                    'url': img['url'],
                    'manufacturer': manufacturer,
                    'manufacturer_index': site_data['manufacturer_index'],
                    'alt_text': img['alt_text'],
                    'source': img['source'],
                    'source_page': img['source_page'],
                    'page_context': img['page_context']
                }

                task = asyncio.create_task(
                    download_single_image(
                        session,
                        img_data,
                        save_path,
                        image_counter[manufacturer]
                    )
                )
                tasks.append(task)

                image_counter[manufacturer] += 1

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r is not None]

