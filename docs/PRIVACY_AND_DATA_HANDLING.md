# Privacy and data handling

## Defaults

- Webcam images are processed locally and are not uploaded or persisted by the Version 2 live loop. No frame recording option is enabled.
- The OCR path captures the desktop into memory and passes it to Tesseract locally. It does not write screenshot files.
- OCR text is not printed and is not returned in default PII detection results. CSV allowlists accept categories, bounding boxes, scores, timings, and trial metadata only. Do not enable `include_text=True` except for synthetic debugging.
- `pii_detection.csv` contains PII category labels and box coordinates; coordinates may reveal where information appears on a screen. Protect the file as potentially sensitive metadata.
- The overlay displays a screenshot in a local OpenCV window. Avoid exposing real personal information during tests; use generated synthetic screens.

## Data collection rules

Use synthetic email addresses, phone-like strings, account/ID-like values, password examples, and addresses. Never use actual credentials, contact data, Aadhaar, payment data, private messages, medical/financial data, or unredacted personal screenshots. Do not commit raw screenshots, camera frames, OCR text, or participant identifiers to version control. The PII manifest evaluator accepts an expected text column solely for synthetic images; its output contains only a recognition correctness flag.

Keep results in access-controlled local storage, encrypt disks/backups where available, define a retention/deletion date before collection, and share only de-identified aggregates. Restrict access to raw experimental logs because timing, trial labels, and screen-region coordinates can still be identifying. Delete data securely when the approved retention period ends. No cloud upload or remote processing is part of this implementation.

## Participants and consent

Obtain informed consent before recording or processing a participant's webcam image, even when frames remain transient/local. Explain camera use, screen capture/OCR, measured variables, retention, access, withdrawal, and whether any screenshots are retained. Obtain institutional ethics/IRB approval or a formal determination where required before human-subject data collection. Store only pseudonymous participant codes; do not collect names, emails, or other direct identifiers in the usability CSV. Free-text comments must not contain identifiers.

## Limitations

Python dependencies or operating-system screen-capture behavior may create transient buffers outside this code's control. The app does not implement encrypted storage, access-control enforcement, consent management, or automated retention deletion. These are operational responsibilities and should be addressed before collecting sensitive data. Do not run the system over a real account screen during research.
