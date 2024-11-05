# process_website.py
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import asyncio
from typing import Dict, List, Set
import re
from aiohttp_retry import RetryClient, ExponentialRetry
from src.utils import setup_logging
import urllib.robotparser
import os

logger = setup_logging(__name__)


def sanitize_filename(filename):
    """Sanitize filename by removing invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def extract_filename(url):
    """Extract filename from URL."""
    match = re.search(r'/([^/]+\.(jpg|jpeg|png|gif|bmp|webp|svg))', url, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return os.path.basename(urlparse(url).path)


def get_sublinks(url, soup):
    """Extract sublinks from soup that belong to the same domain."""
    sublinks = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(url, href)
        if full_url.startswith(url):
            sublinks.add(full_url)
    return sublinks


def extract_menu_items(soup):
    """Extract menu items from navigation elements."""
    menu_items = []
    nav_elements = soup.find_all(['nav', 'ul', 'div'], class_=lambda x: x and 'menu' in x.lower())
    for nav in nav_elements:
        items = nav.find_all('a')
        for item in items:
            menu_items.append({
                'text': item.get_text(strip=True),
                'url': item.get('href')
            })
    return menu_items


class WebsiteProcessor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.visited_urls = set()
        self.downloaded_images = set()
        self.max_pages_per_domain = 500
        self.delay = 1
        self.rp = None

    async def fetch_page(self, session: aiohttp.ClientSession, url: str, retries: int = 3) -> str:
        """Fetch page content with retry mechanism."""
        retry_options = ExponentialRetry(attempts=retries)
        retry_client = RetryClient(client_session=session, retry_options=retry_options)
        try:
            async with retry_client.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
        return None

    def extract_images(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Extract images from both img tags and CSS backgrounds."""
        images = []
        page_content = soup.get_text()

        # Extract from img tags
        for img in soup.find_all('img'):
            img_url = img.get('data-srclazy') or img.get('src')
            if img_url:
                img_url = urljoin(url, img_url)
                if img_url not in self.downloaded_images:
                    images.append({
                        'url': img_url,
                        'alt_text': img.get('alt', 'N/A'),
                        'source': 'img_tag',
                        'source_page': url,
                        'page_context': page_content
                    })
                    self.downloaded_images.add(img_url)

        # Extract from style tags
        url_pattern = re.compile(r'background-image:\s*url\((.*?)\)')
        for style_tag in soup.find_all('style'):
            style_content = style_tag.string
            if style_content:
                urls = url_pattern.findall(style_content)
                for bg_url in urls:
                    clean_url = urljoin(url, bg_url.strip("'\""))
                    if not clean_url.startswith('data:') and clean_url not in self.downloaded_images:
                        images.append({
                            'url': clean_url,
                            'alt_text': 'N/A',
                            'source': 'style_tag',
                            'source_page': url,
                            'page_context': page_content
                        })
                        self.downloaded_images.add(clean_url)

        # Extract from inline styles
        for element in soup.find_all(style=True):
            style = element.get('style', '')
            if 'background-image' in style:
                urls = url_pattern.findall(style)
                for bg_url in urls:
                    clean_url = urljoin(url, bg_url.strip("'\""))
                    if not clean_url.startswith('data:') and clean_url not in self.downloaded_images:
                        images.append({
                            'url': clean_url,
                            'alt_text': 'N/A',
                            'source': 'inline_style',
                            'source_page': url,
                            'page_context': page_content
                        })
                        self.downloaded_images.add(clean_url)

        return images

    async def process_url(self, session: aiohttp.ClientSession, url: str, domain: str,
                          is_menu_item: bool, processed_data: Dict):
        """Process a single URL and extract content."""
        if url in self.visited_urls or not self.rp.can_fetch("*", url):
            return

        self.visited_urls.add(url)
        logger.info(f"Processing {'menu item' if is_menu_item else 'sublink'}: {url}")

        html = await self.fetch_page(session, url)
        if not html:
            logger.error(f"Failed to fetch page: {url}")
            return

        soup = BeautifulSoup(html, 'html.parser')
        page_content = soup.get_text(separator=' ', strip=True)
        images = self.extract_images(soup, url)

        processed_data['page_contexts'][url] = page_content
        processed_data['images'].extend(images)

        # Get sublinks and process them
        sublinks = get_sublinks(url, soup)
        for sublink in sublinks:
            if sublink not in self.visited_urls and self.rp.can_fetch("*", sublink):
                await self.process_url(session, sublink, domain, False, processed_data)
                await asyncio.sleep(self.delay)

    async def process_domain(self, session: aiohttp.ClientSession, manufacturer: str,
                             website_url: str, manufacturer_index: int) -> Dict:
        """Process an entire domain including all sublinks."""
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'http://' + website_url

        # Set up robots.txt parser
        self.rp = urllib.robotparser.RobotFileParser()
        robots_url = urljoin(website_url, '/robots.txt')
        try:
            async with session.get(robots_url) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    self.rp.parse(robots_content.splitlines())
                else:
                    self.rp = None
        except Exception as e:
            logger.error(f"Error setting up robots parser for {website_url}: {str(e)}")
            self.rp = None
            return None

        processed_data = {
            'manufacturer': manufacturer,
            'manufacturer_index': manufacturer_index,
            'website_url': website_url,
            'images': [],
            'page_contexts': {}
        }

        # Fetch and process the home page
        html = await self.fetch_page(session, website_url)
        if not html:
            logger.error(f"Failed to fetch home page: {website_url}")
            return None

        soup = BeautifulSoup(html, 'html.parser')
        menu_items = extract_menu_items(soup)

        # Process home page
        await self.process_url(session, website_url, website_url, False, processed_data)

        # Process menu items and their sublinks
        for item in menu_items:
            item_url = urljoin(website_url, item['url'])
            await self.process_url(session, item_url, website_url, True, processed_data)
            await asyncio.sleep(self.delay)

        logger.info(f"Completed processing domain {website_url}. "
                    f"Processed {len(self.visited_urls)} pages, "
                    f"found {len(processed_data['images'])} images")

        return processed_data


async def process_websites(manufacturer_data: Dict[str, Dict]) -> List[Dict]:
    """Process all websites."""
    processor = WebsiteProcessor()

    async with aiohttp.ClientSession(headers=processor.headers) as session:
        tasks = []
        for idx, (manufacturer, data) in enumerate(manufacturer_data.items()):
            website_url = data.get('Website')
            if website_url:
                task = asyncio.create_task(
                    processor.process_domain(session, manufacturer, website_url, idx)
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r is not None]