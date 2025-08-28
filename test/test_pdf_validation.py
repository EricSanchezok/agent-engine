#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for PDF validation functionality in ArxivFetcher
"""

import asyncio
import base64
import io
import tempfile
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.arxiv.arxiv import ArXivFetcher

def create_test_pdf(content: bytes, filename: str = None) -> str:
    """Create a test PDF file and return its base64 encoding"""
    if filename is None:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(content)
            filename = f.name
    
    # Read and encode to base64
    with open(filename, 'rb') as f:
        pdf_bytes = f.read()
    
    # Clean up temp file
    if filename.startswith(tempfile.gettempdir()):
        os.unlink(filename)
    
    return base64.b64encode(pdf_bytes).decode('ascii')

def test_pdf_validation():
    """Test the PDF validation functionality"""
    fetcher = ArXivFetcher()
    
    print("Testing PDF validation functionality...")
    
    # Test 1: Valid PDF
    print("\n1. Testing valid PDF...")
    valid_pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n364\n%%EOF'
    valid_pdf_b64 = create_test_pdf(valid_pdf_content)
    result = fetcher._validate_pdf_integrity(valid_pdf_b64)
    print(f"Valid PDF validation result: {result}")
    assert result == True, "Valid PDF should pass validation"
    
    # Test 2: Incomplete PDF (missing EOF marker)
    print("\n2. Testing incomplete PDF (missing EOF marker)...")
    incomplete_pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n364'
    incomplete_pdf_b64 = create_test_pdf(incomplete_pdf_content)
    result = fetcher._validate_pdf_integrity(incomplete_pdf_b64)
    print(f"Incomplete PDF validation result: {result}")
    assert result == False, "Incomplete PDF should fail validation"
    
    # Test 3: Invalid PDF header
    print("\n3. Testing invalid PDF header...")
    invalid_header_content = b'NOT_A_PDF\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF'
    invalid_header_b64 = create_test_pdf(invalid_header_content)
    result = fetcher._validate_pdf_integrity(invalid_header_b64)
    print(f"Invalid header PDF validation result: {result}")
    assert result == False, "Invalid header PDF should fail validation"
    
    # Test 4: Empty PDF
    print("\n4. Testing empty PDF...")
    empty_pdf_b64 = base64.b64encode(b'').decode('ascii')
    result = fetcher._validate_pdf_integrity(empty_pdf_b64)
    print(f"Empty PDF validation result: {result}")
    assert result == False, "Empty PDF should fail validation"
    
    # Test 5: Corrupted PDF (valid header but corrupted content)
    print("\n5. Testing corrupted PDF...")
    corrupted_pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000206 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n364\n%%EOF\n%%EOF\n%%EOF'
    corrupted_pdf_b64 = create_test_pdf(corrupted_pdf_content)
    result = fetcher._validate_pdf_integrity(corrupted_pdf_b64)
    print(f"Corrupted PDF validation result: {result}")
    # This might pass basic validation but fail PyPDF2 validation
    
    print("\n✅ All PDF validation tests completed successfully!")

if __name__ == "__main__":
    test_pdf_validation()
