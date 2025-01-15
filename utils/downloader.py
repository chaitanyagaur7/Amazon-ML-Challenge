import os
import logging
import urllib.request
from utils.file_utils import create_directory
import time 
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
def download_image(image_link, save_folder, retries=3, delay=3):
    # Ensure the save folder exists
    create_directory(save_folder)

    if not isinstance(image_link, str):
        logging.warning(f"Invalid image link: {image_link}")
        return None

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        logging.info(f"Image already exists at {image_save_path}, skipping download.")
        return image_save_path

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            logging.info(f"Downloaded image: {image_save_path}")
            return image_save_path
        except Exception as e:
            logging.warning(f"Failed to download {image_link} on attempt {attempt + 1}. Error: {e}")
            time.sleep(delay)

    logging.error(f"Failed to download image after {retries} attempts: {image_link}")
    return None

# Optimized concurrent image download
def download_images_concurrently(image_links, save_folder):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_image, link, save_folder): link for link in image_links}
        for future in as_completed(futures):
            link = futures[future]
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Image download failed for {link}: {exc}")