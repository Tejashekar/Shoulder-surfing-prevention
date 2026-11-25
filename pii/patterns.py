"""Regex PII categories applied to space-joined OCR tokens.

Design notes (v2.1):
- `aadhaar` requires the printed 4-4-4 grouping (space or hyphen separators) so
  that an unformatted 12-digit account number is not also reported as Aadhaar.
- `card` requires four 4-digit groups; it is tried before `aadhaar` in the
  detector so a 16-digit card is not reported twice.
- `phone` requires a 10-digit national number (3-3-4 or 5-5) with an optional
  country code; the earlier pattern accepted 7-digit fragments and matched
  timestamps, build numbers, and window coordinates on live screens.
- `email` tolerates a single space after a dot inside the domain because
  Tesseract frequently splits `user@example.` and `com` into two tokens.
- Keyword-anchored categories (`password`, `account`) expose the sensitive
  span as a named group `value`; only that span is boxed/redacted.
- `account` tolerates an OCR-split "Account number : 1234" (keyword, optional
  qualifier, optional punctuation, whitespace) instead of a single separator.
"""

PATTERNS = {
    "card": r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b",
    "aadhaar": r"\b\d{4}[- ]\d{4}[- ]\d{4}\b(?![- ]?\d)",
    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9-]+(?:\.\s?[A-Z0-9-]+)*\.\s?[A-Z]{2,}\b",
    "phone": (
        r"(?<![\d.:/-])"
        r"(?:\+?\d{1,3}[ -]?)?"
        r"(?:\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}|\d{5}[ -]\d{5})"
        r"(?![\d.:/-])"
    ),
    "password": r"\b(?:password|passcode|pin)\s*[:=]\s*(?P<value>\S+)",
    "account": r"\b(?:account|acct|a/c)\s*(?:no\.?|number|num|#)?\s*[:=#-]?\s*(?P<value>\d{6,18})\b",
    "address": r"\b\d{1,5}\s+[A-Z0-9 .'-]{2,40}\s(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|cross|main)\b",
}

# Categories whose match should suppress a weaker category on the same span.
SUPPRESSES = {
    "card": ("aadhaar", "phone", "account"),
    "account": ("aadhaar", "phone"),
    "aadhaar": ("phone",),
}
