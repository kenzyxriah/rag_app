from docling.document_converter import DocumentConverter # there's a library does this exactly
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

async def parse(source: Path, use_ocr: bool = False):

    if source.name.endswith('.txt'):
        with open(source, 'r', encoding='utf-8') as f:
            return f.read()
        
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = use_ocr  
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(source)
        return result.document.export_to_markdown()

    except Exception as e:
        raise RuntimeError(f"Failed to parse file: {e}")
