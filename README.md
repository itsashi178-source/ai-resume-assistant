# 📄 AI Resume ATS Analyzer

A Streamlit app that analyzes a resume (PDF or DOCX) with Google's Gemini Flash model and returns:

- An estimated **ATS score** (0–100)
- Score breakdown (formatting, keywords, experience, skills, education, readability)
- Matched and missing **keywords** (optionally against a pasted job description)
- Strengths, weaknesses and prioritized **improvements**
- Suggested **bullet-point rewrites**
- Downloadable JSON report

> The score is an AI-based estimate, not the output of a real ATS. Use it as guidance.
> Resume text is sent to Google's Gemini API for analysis.

## Run locally

```bash
git clone https://github.com/<your-username>/resume-ats-analyzer.git
cd resume-ats-analyzer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at <https://aistudio.google.com/apikey>, then either:

- paste it into the app's sidebar, **or**
- create `.streamlit/secrets.toml`:
  ```toml
  GEMINI_API_KEY = "your-key-here"
  ```
- or set an environment variable: `export GEMINI_API_KEY=your-key-here`

Start the app:

```bash
streamlit run app.py
```

## Configuration

The model defaults to `gemini-3.5-flash`. You can change it in the sidebar, or edit `DEFAULT_MODEL` in `app.py` if Google renames or retires it.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit your API key).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Project structure

```
app.py             # Streamlit UI + Gemini logic
requirements.txt   # Python dependencies
README.md
.gitignore
```

## Limitations

- Scanned/image-only PDFs can't be read (no OCR).
- Very long resumes are truncated to ~20,000 characters.
- Free Gemini API tiers have rate limits.
