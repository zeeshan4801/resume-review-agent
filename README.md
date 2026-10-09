
# AI Resume Review Agent

A beginner-friendly resume review application built using:

- Python 3.11
- Streamlit
- CrewAI
- Groq
- pypdf

## Overview

This application compares a candidate's resume with
a target job description.

It uses exactly:
- One CrewAI Agent
- One CrewAI Task
- One CrewAI Crew

No database, RAG, authentication, or Docker is required.

## Features

- Paste resume text
- Upload a PDF resume
- Paste a job description
- Analyze demonstrated skills
- Identify explicitly missing requirements
- Identify unknown or undemonstrated requirements
- Review experience and education gaps
- Suggest truthful resume improvements
- Suggest relevant keywords
- Generate a prioritized action plan

## Accuracy Rules

The application must not invent candidate details.

A qualification is demonstrated only when supported
by resume evidence.

Unmentioned qualifications are classified as
unknown or not demonstrated.

AI assessments should be manually verified.

## Project Structure

resume-review-agent/
    app.py
    requirements.txt
    README.md
    .gitignore
    .streamlit/
        secrets.toml.example

## Requirements

Python 3.11 and a Groq API key.

## Local Installation

### Step 1: Clone repository

git clone https://github.com/YOUR_USERNAME/resume-review-agent.git

cd resume-review-agent

### Step 2: Create virtual environment

Windows:

py -3.11 -m venv .venv

.venv\Scripts\activate

macOS/Linux:

python3.11 -m venv .venv

source .venv/bin/activate

### Step 3: Install dependencies

python -m pip install --upgrade pip

pip install -r requirements.txt

### Step 4: Configure secrets

Create:

.streamlit/secrets.toml

Add:

GROQ_API_KEY = "your_actual_api_key"

GROQ_MODEL = "openai/gpt-oss-120b"

Do not commit secrets.toml.

### Step 5: Run application

streamlit run app.py

Open the local URL displayed in the terminal.

## Deployment

1. Upload the project to GitHub.
2. Open https://share.streamlit.io
3. Select Create app.
4. Choose your GitHub repository.
5. Select the main branch.
6. Set the entrypoint to app.py.
7. Open Advanced settings.
8. Select Python 3.11.
9. Add GROQ_API_KEY and GROQ_MODEL as secrets.
10. Deploy the application.

## Privacy

The application does not intentionally persist
resume or job description data.

Inputs are sent to the configured Groq model
for analysis.

Groq and hosting-provider policies apply.

Do not upload sensitive personal information
without appropriate permission.

## PDF Limitations

Text-based PDFs are supported.

Scanned PDFs without extractable text are not
supported because OCR is not included.

Password-protected PDFs are not supported.

## Troubleshooting

Missing API key:
Check Streamlit Secrets.

Invalid PDF:
Upload a text-based PDF or paste resume text.

Model unavailable:
Check GROQ_MODEL and Groq model availability.

Rate limit:
Wait and retry.

Dependency installation failure:
Verify Python 3.11 and compatible package versions.

## Disclaimer

This application provides AI-generated resume feedback.

It does not guarantee employment, hiring eligibility,
or factual correctness.

Always verify the analysis before using it.
