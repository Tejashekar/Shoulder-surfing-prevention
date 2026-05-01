# Shoulder Surfing Prevention

A system designed to detect and prevent shoulder surfing attacks using face detection, gaze estimation, and PII detection.

## Project Structure

```
├── detection/          # Face detection and gaze estimation modules
├── overlay/            # Overlay engine for visual feedback
├── pii/                # PII detection and OCR engine
├── risk/               # Risk assessment engine
├── my-project/         # Web frontend (TypeScript/Node.js)
├── main.py             # Main entry point
└── test_*.py           # Test files
```

## Features

- Face detection and gaze estimation
- Proximity monitoring for unauthorized shoulder surfing
- PII detection through OCR
- Risk scoring and assessment
- Web-based overlay interface

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python main.py
```

## Testing

Run tests with:
```bash
python test_detection.py
python test_ocr_basic.py
python test_pii.py
```

## License

[Add your license here]
