from pypdf import PdfReader

pdf_path = "/home/gurpreet/Desktop/Spring2025/CSCI115/Project/e115_SMART/data/pdfs/CSCI_83/Bayes MCMC Models.pdf"

# Try to read the PDF
try:
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    print("PDF is readable. Extracted text preview:\n", text[:500])  # Print first 500 characters
except Exception as e:
    print("Error reading PDF:", e)
