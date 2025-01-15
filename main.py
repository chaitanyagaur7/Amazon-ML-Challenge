import logging
from utils.downloader import download_images_concurrently
from utils.ocr import extract_text_from_image
from utils.ner import extract_entities
from utils.regex_extractor import extract_features
from utils.file_utils import create_directory, load_csv, save_csv
from config.settings import TEST_CSV_PATH, SAVE_FOLDER, OUTPUT_PATH

logging.info("Starting the program...")

# Load test data
test_df = load_csv(TEST_CSV_PATH)

# Download images
download_images_concurrently(test_df['image_link'], SAVE_FOLDER)

# Process images and extract features
results = []
for _, row in test_df.iterrows():
    image_path = f"{SAVE_FOLDER}/{row['image_link'].split('/')[-1]}"
    text = extract_text_from_image(image_path)
    ner_entities = extract_entities(text)
    regex_entities = extract_features(text)
    results.append({"index": row['index'], "text": text, "ner": ner_entities, "regex": regex_entities})

# Save results
save_csv(results, OUTPUT_PATH)
logging.info(f"Results saved to {OUTPUT_PATH}")
