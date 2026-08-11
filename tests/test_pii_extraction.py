"""
tests/test_pii_extraction.py
==============================
Tests for PII extraction attack and the built-in regex fallback scanner.
"""
import pytest
from src.attacks.pii_extraction import _fallback_pii_scan

class TestFallbackPIIScanner:
    def test_detects_aadhaar_number_in_response(self):
        text = "Your Aadhaar is 2345 6789 0123."
        found = _fallback_pii_scan(text)
        types = [t for t, _ in found]
        assert "AADHAAR" in types

    def test_detects_pan_card_format(self):
        text = "PAN: ABCDE1234F"
        found = _fallback_pii_scan(text)
        types = [t for t, _ in found]
        assert "PAN" in types

    def test_detects_indian_mobile_number_with_country_code(self):
        text = "Call +91-9876543210 for help."
        found = _fallback_pii_scan(text)
        types = [t for t, _ in found]
        assert "MOBILE" in types

    def test_detects_email_address(self):
        text = "Send it to user@example.com"
        found = _fallback_pii_scan(text)
        types = [t for t, _ in found]
        assert "EMAIL" in types

    def test_returns_empty_for_clean_text(self):
        text = "India has a population of 1.4 billion people."
        found = _fallback_pii_scan(text)
        pii_types = [t for t, _ in found if t not in ("BANK_ACCOUNT_CANDIDATE",)]
        assert len(pii_types) == 0

    def test_aadhaar_without_spaces_detected(self):
        text = "Aadhaar: 234567890123"
        found = _fallback_pii_scan(text)
        types = [t for t, _ in found]
        assert "AADHAAR" in types

    def test_multiple_pii_types_in_one_response(self):
        text = "Name: John, PAN: ABCDE1234F, Mobile: +91-9876543210"
        found = _fallback_pii_scan(text)
        types = {t for t, _ in found}
        assert "PAN" in types
        assert "MOBILE" in types
