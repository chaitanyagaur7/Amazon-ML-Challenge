import re

entity_patterns = {
    "item_weight": re.compile(r'(\d+(\.\d+)?\s?(gram|kilogram|microgram|milligram|ounce|pound|ton))', re.IGNORECASE),
    "item_volume": re.compile(r'(\d+(\.\d+)?\s?(centilitre|cubic foot|cubic inch|cup|decilitre|fluid ounce|gallon|imperial gallon|litre|microlitre|millilitre|pint|quart))', re.IGNORECASE),
    "height": re.compile(r'(\d+(\.\d+)?\s?(centimetre|foot|inch|metre|millimetre|yard))', re.IGNORECASE),
    "width": re.compile(r'(\d+(\.\d+)?\s?(centimetre|foot|inch|metre|millimetre|yard))', re.IGNORECASE),
    "dimension": re.compile(r'(\d+(\.\d+)?\s?(mm|cm|m|in|ft|yd))', re.IGNORECASE),
    "voltage": re.compile(r'(\d+(\.\d+)?\s?(volt|kilovolt|millivolt))', re.IGNORECASE),
    "wattage": re.compile(r'(\d+(\.\d+)?\s?(watt|kilowatt))', re.IGNORECASE)
}


def extract_features(text):
    entity_features = {}
    for name, pattern in entity_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            entity_features[name] = [match[0] for match in matches]
    return entity_features
