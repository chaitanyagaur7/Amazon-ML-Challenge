import easyocr

reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path):
    results = reader.readtext(image_path)
    return ' '.join([result[1] for result in results])
