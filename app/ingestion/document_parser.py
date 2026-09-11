# Document AI resume extraction
from google.cloud import documentai_v1 as documentai
from app.config import DOCAI_PROJECT_ID, LOCATION, DOCAI_PROCESSOR_ID

def parse_resume_pdf(file_bytes: bytes, mime_type: str = "application/pdf") -> str:
    """Extracts raw text from PDF resume using Document AI OCR."""
    if not DOCAI_PROCESSOR_ID:
        return "Document AI Processor ID not set. Resume stored as raw binary."
    
    try:
        client = documentai.DocumentProcessorServiceClient()
        processor_name = client.processor_path(DOCAI_PROJECT_ID, LOCATION, DOCAI_PROCESSOR_ID)
        
        raw_document = documentai.RawDocument(content=file_bytes, mime_type=mime_type)
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        result = client.process_document(request=request)
        return result.document.text
    except Exception as e:
        return f"Error extracting resume text: {str(e)}"