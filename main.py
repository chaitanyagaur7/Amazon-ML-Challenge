import os
import pandas as pd
import torch
import easyocr  # Deep learning-based OCR
from transformers import pipeline  # NER from Transformers
import re  # Regular expressions for pattern matching
from pathlib import Path

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Initialize reader for English language

# Initialize NER Pipeline using DistilBERT
device = -1  # Let the model automatically choose GPU or CPU
ner = pipeline("ner", model="distilbert-base-uncased", device=device)

# Correct paths for the CSV files
train_csv_file_path = r'/kaggle/input/dataset/train.csv'
test_csv_file_path = r'/kaggle/input/dataset/test.csv'

# Load the Data
print("Loading training and testing data from CSV files.")
train_df = pd.read_csv(train_csv_file_path)
test_df = pd.read_csv(test_csv_file_path)

# Function to create directory if it does not exist
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

# Batch OCR Text Extraction
def batch_extract_text_from_images(image_paths, batch_size=10):
    batched_results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        results = [reader.readtext(image_path) for image_path in batch_paths]
        batched_results.extend(results)
    return batched_results

# Regex patterns for extracting entity features
entity_patterns = {
    "item_weight": r'(\d+(\.\d+)?\s?(gram|kilogram|microgram|milligram|ounce|pound|ton|g|kg|mg|lb))',
    "item_volume": r'(\d+(\.\d+)?\s?(centilitre|cubic foot|cubic inch|cup|decilitre|fluid ounce|gallon|imperial gallon|litre|microlitre|millilitre|pint|quart|cl|ml|l|fl oz|gal))',
    "height": r'(\d+(\.\d+)?\s?(centimetre|foot|inch|metre|millimetre|yard|cm|ft|in|mm|m|yd))',
    "width": r'(\d+(\.\d+)?\s?(centimetre|foot|inch|metre|millimetre|yard|cm|ft|in|mm|m|yd))',
    "depth": r'(\d+(\.\d+)?\s?(centimetre|foot|inch|metre|millimetre|yard|cm|ft|in|mm|m|yd))',
    "dimension": r'(\d+(\.\d+)?\s?(mm|cm|m|in|ft|yd))',
    "voltage": r'(\d+(\.\d+)?\s?(volt|kilovolt|millivolt|V|kV|mV))',
    "wattage": r'(\d+(\.\d+)?\s?(watt|kilowatt|W|kW))',
    "maximum_weight_recommendation": r'(\d+(\.\d+)?\s?(gram|kilogram|microgram|milligram|ounce|pound|ton|g|kg|mg|lb))'
}

# Function to extract entities from text using NER (batch processing)
def batch_extract_entities(texts, batch_size=10):
    batched_entities = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_entities = ner(batch_texts)
        batched_entities.extend(batch_entities)
    return batched_entities

# Combine text extraction and entity classification
def extract_features_from_image(image_paths, batch_size=10):
    # Batch process OCR
    extracted_texts = batch_extract_text_from_images(image_paths, batch_size)
    
    # Flatten the OCR results
    combined_texts = [' '.join([result[1] for result in batch]) for batch in extracted_texts]
    
    # Batch process NER
    batched_entities = batch_extract_entities(combined_texts, batch_size)

    entity_features_list = []
    for text in combined_texts:
        entity_features = {}
        for entity_name, pattern in entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entity_features[entity_name] = [match[0] for match in matches]
        entity_features_list.append(entity_features)

    return combined_texts, entity_features_list

# Predict for Test Data
def generate_predictions(test_df, save_folder, batch_size=10):
    # Use image paths from the folder instead of links
    image_links = test_df['image_link'].unique()

    # Generate image paths based on existing images
    image_paths = [os.path.join(save_folder, Path(link).name) for link in image_links if os.path.exists(os.path.join(save_folder, Path(link).name))]

    # Check if any images were found
    if not image_paths:
        print("No images found for the provided paths.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Extract features in batches
    print(f"Processing images in batches of {batch_size}...")
    texts, extracted_entities = extract_features_from_image(image_paths, batch_size)

    # Ensure the iteration only goes over the processed data length
    num_processed = min(len(texts), len(extracted_entities), len(test_df))

    # Store results in dataframe
    results = []
    for i, row in enumerate(test_df.itertuples(index=True)):
        if i >= num_processed:
            break
        entity_name = row.entity_name  # Assuming this column exists in test_df

        results.append({
            "index": row.Index,  # Use row.Index which is from itertuples
            "text": texts[i],
            "extracted_entities": extracted_entities[i]
        })

    return pd.DataFrame(results)

# Run the prediction generation
print("Starting prediction generation for test data.")
save_folder = r'/content/Someresults'
create_directory(save_folder)  # Ensure save folder is created
predictions_df = generate_predictions(test_df, save_folder, batch_size=10)
predictions_df.to_csv('output_advanced_features1.csv', index=False)
print("Predictions saved to output_advanced_features1.csv")

print("Completed processing and saved the predictions.")
