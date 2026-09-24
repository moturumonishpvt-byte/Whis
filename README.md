# WHIS

A long-term personal AI computer assistant.

Current development status:
Phase 1 — Development Environment & Project Foundation

## Requirements

- Python 3.13.x

## Virtual Environment Setup

To set up the virtual environment:
```bash
python -m venv .venv
```

To activate the virtual environment:
- Windows: `.venv\Scripts\activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running WHIS

```bash
python -m app.main
```

## Running Tests

```bash
python -m unittest discover tests
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values.

**Security Note:** `.env` must never be committed to version control. Keep your API keys and secrets safe!

## Project Structure

- `app/`: Core application logic
- `config/`: Configuration settings and logging
- `tests/`: Unit tests
- `scripts/`: Helper scripts
- `logs/`: Application logs
