# Contributing to Sports Betting ML

Welcome! This project aims to build machine learning models for sports betting predictions.

## How to Contribute

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Make** your changes
4. **Test** locally before submitting
5. **Submit** a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/sports-betting-ml.git
cd sports-betting-ml

# Create virtual environment (Python 3.12+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest ruff

# Run the app
streamlit run app.py
```

## Tests and lint

```bash
python -m pytest tests/ -v
ruff check . --select E,F,W --ignore E501
```

`pytest.ini` sets `pythonpath = .` so tests import `utils`, `model`, and `data` without `sys.path` hacks.

## Code Style

- Follow PEP 8 for Python
- Use meaningful variable names
- Comment complex logic
- This repo is **NBA-focused**. Do not add NFL/MLB/NHL without a real data source and evaluation path.

## Issues

Check the Issues tab for good first contributions:
- Improve tests and evaluation
- Wire real historical stats into training (currently synthetic in `model/train.py`)
- Docs and packaging fixes

## Questions?

Open an issue for questions about contributing!
