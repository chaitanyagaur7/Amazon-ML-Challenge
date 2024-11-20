# Advanced OCR and Entity Extraction Pipeline

This repository demonstrates a deep learning-based pipeline for **Optical Character Recognition (OCR)** and **Named Entity Recognition (NER)** using Python libraries like EasyOCR and Huggingface Transformers. It includes functionality for extracting text from images, identifying specific entities using regular expressions, and saving the extracted features into a structured CSV format.

## Features

- Batch processing for OCR using **EasyOCR**.
- Named Entity Recognition using **DistilBERT** from Huggingface Transformers.
- Regular expressions for extracting specific entity features such as dimensions, voltage, wattage, etc.
- Automatic directory management for output storage.
- CSV file handling for structured input and output.

---

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
  - [1. OCR](#1-ocr)
  - [2. NER](#2-ner)
  - [3. Entity Extraction](#3-entity-extraction)
- [Outputs](#outputs)

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your_username/ocr-ner-pipeline.git
cd ocr-ner-pipeline
pip install -r requirements.txt
```

## Dependencies 

- Python 3.8+
- EasyOCR: Deep learning-based OCR.
- Transformers: Huggingface library for NLP tasks.
- Pandas: Data manipulation and CSV handling.
- PyTorch: Required for Transformers and EasyOCR.
- Regex: For custom entity extraction patterns.

 ```python
  pip install easyocr transformers torch pandas
 ```


