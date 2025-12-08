"""
Test PDF extraction
"""
from pathlib import Path

# Test extraction
pdf_path = "./data/documents/How to believe in God even when the world sucks with Nadia Bolz-Weber.pdf"

print("Testing PDF extraction...")

# Try pdfplumber
try:
    import pdfplumber
    print("\n--- pdfplumber ---")
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages[:3]):
            text = page.extract_text() or ""
            print(f"Page {i+1}: {len(text)} chars")
            if text:
                print(f"  Preview: {text[:200]}...")
except Exception as e:
    print(f"pdfplumber failed: {e}")

# Try PyPDF2
try:
    from PyPDF2 import PdfReader
    print("\n--- PyPDF2 ---")
    reader = PdfReader(pdf_path)
    print(f"Pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages[:3]):
        text = page.extract_text() or ""
        print(f"Page {i+1}: {len(text)} chars")
        if text:
            print(f"  Preview: {text[:200]}...")
except Exception as e:
    print(f"PyPDF2 failed: {e}")

print("\n✅ Test complete")