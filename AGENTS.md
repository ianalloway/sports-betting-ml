# AGENTS.md - Sports Betting ML

## Overview
Machine learning platform for sports betting analytics. Predictive models for NFL, NBA, MLB, NHL with data pipelines and a Streamlit dashboard.

## Tech Stack
- **Language:** Python 3
- **Dependencies:** See requirements.txt
- **Containerization:** Docker + docker-compose

## Commands
```bash
pip install -r requirements.txt    # Install dependencies
python app.py                      # Run the Streamlit app
docker-compose up                  # Run via Docker
```

## Project Structure
```
app.py               # Main Streamlit application
data/                # Training/test data
model/               # ML model definitions and saved models
utils/               # Utility functions
docs/                # Documentation
demo.gif             # Demo recording
Dockerfile           # Container definition
docker-compose.yml   # Multi-container setup
env.example         # Environment variable template
```

## Key Conventions
- Related to ai-advantage repo (frontend) and openclaw-skills/sports-odds (OpenClaw skill)
- Uses .env for API keys (copy env.example)
- Has GitHub Actions CI (.github/)

## Owner
Ian Alloway (@ianalloway) - Data Scientist specializing in sports analytics and ML.
