# Security Policy

## Supported Versions

This repository is a public training / demo app. Use the latest `main` commit.

## Reporting a Vulnerability

Please report security issues privately to **ian@allowayllc.com**.

Include:

- a description of the issue and impact
- steps to reproduce (or a proof of concept)
- affected commit / release if known

Do **not** open a public GitHub issue for vulnerabilities that could expose API keys, enable remote code execution, or leak user data.

## Notes for Contributors

- Never commit `.env`, API keys, or trained artifacts that embed secrets
- Prefer the Odds API key via environment variables (`ODDS_API_KEY`)
- Model artifacts under `model/artifacts/` are generated locally / in CI and should stay out of git
