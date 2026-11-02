"""AI Resume ATS Analyzer - Streamlit + Google Gemini Flash.

Upload a resume (PDF or DOCX), optionally paste a job description, and get:
  * an estimated ATS score (0-100) with a section-wise breakdown
  * keyword analysis (matched / missing)
  * strengths, weaknesses and concrete improvements
  * rewritten bullet-point suggestions
"""

import io
import json
import os
import re

import streamlit as st
from docx import Document
from pypdf import PdfReader

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "gemini-3.5-flash"  # change here (or in the sidebar) if Google renames it
MAX_FILE_MB = 5
MAX_RESUME_CHARS = 20000
MAX_JD_CHARS = 8000

SCORE_SECTIONS = {
    "formatting": "Formatting & ATS-friendliness",
    "keywords": "Keywords & relevance",
    "experience": "Experience & achievements",
    "skills": "Skills",
    "education": "Education",
    "readability": "Clarity & readability",
}

SYSTEM_PROMPT = """You are an expert technical recruiter and Applicant Tracking System (ATS) analyst.
You evaluate resumes strictly and honestly. Do not inflate scores.
Base every statement ONLY on the resume text provided. Never invent experience, employers, or numbers.
Respond with a single valid JSON object and nothing else."""


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password-protected. Please upload an unlocked copy.")
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_text_from_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_resume_text(filename: str, data: bytes) -> str:
    """Return cleaned text from a PDF/DOCX. Raises ValueError with a friendly message."""
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            raw = extract_text_from_pdf(data)
        elif name.endswith(".docx"):
            raw = extract_text_from_docx(data)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or DOCX.")
    except ValueError:
        raise
    except Exception as exc:  # corrupted / unreadable file
        raise ValueError(f"Could not read the file ({type(exc).__name__}). Is it corrupted?") from exc

    text = clean_text(raw)
    if len(text) < 50:
        raise ValueError(
            "Almost no text could be extracted. The resume may be a scanned image, "
            "which is also a problem for real ATS systems. Export it as a text-based PDF or DOCX."
        )
    return text


# --------------------------------------------------------------------------- #
# Prompt + response handling
# --------------------------------------------------------------------------- #
def build_prompt(resume_text: str, job_description: str) -> str:
    jd_block = (
        f"JOB DESCRIPTION:\n\"\"\"\n{job_description[:MAX_JD_CHARS]}\n\"\"\"\n"
        "Score keyword match and relevance against this job description."
        if job_description.strip()
        else "No job description was provided. Evaluate general ATS-readiness and "
        "infer the most likely target role from the resume."
    )

    return f"""Analyze the resume below for ATS compatibility.

RESUME:
\"\"\"
{resume_text[:MAX_RESUME_CHARS]}
\"\"\"

{jd_block}

Return JSON with EXACTLY this structure:
{{
  "ats_score": <integer 0-100>,
  "summary": "<2-3 sentence overall verdict>",
  "detected_role": "<target role you inferred or were given>",
  "section_scores": {{
    "formatting": <0-100>, "keywords": <0-100>, "experience": <0-100>,
    "skills": <0-100>, "education": <0-100>, "readability": <0-100>
  }},
  "matched_keywords": ["..."],
  "missing_keywords": ["..."],
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvements": [
    {{"priority": "High|Medium|Low", "area": "<short label>", "suggestion": "<specific, actionable fix>"}}
  ],
  "bullet_rewrites": [
    {{"original": "<exact bullet from the resume>", "improved": "<stronger version using action verb + measurable impact, without inventing facts>"}}
  ],
  "missing_sections": ["<standard sections absent from the resume, e.g. Summary, Projects>"]
}}

Rules:
- Give 5-8 improvements, ordered by priority.
- Give 3-5 bullet_rewrites using bullets that really appear in the resume. If you must add a number, use a placeholder like [X%].
- Max 15 items each in matched_keywords and missing_keywords.
- ats_score must reflect formatting, keyword relevance, quantified achievements, section completeness and clarity.
"""


def parse_json_response(raw: str) -> dict:
    """Parse model output into a dict, tolerating code fences / stray text."""
    if not raw or not raw.strip():
        raise ValueError("The AI returned an empty response. Please try again.")
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The AI response was not valid JSON. Please try again.")
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("The AI response was not valid JSON. Please try again.") from exc
    if not isinstance(data, dict):
        raise ValueError("Unexpected AI response format. Please try again.")
    return data


def _clamp_score(value, default=0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _str_list(value) -> list:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def normalize_result(data: dict) -> dict:
    """Make sure every field exists and has the right type so the UI never crashes."""
    scores_in = data.get("section_scores") if isinstance(data.get("section_scores"), dict) else {}
    improvements = []
    for item in data.get("improvements") or []:
        if isinstance(item, dict):
            priority = str(item.get("priority", "Medium")).strip().capitalize()
            if priority not in ("High", "Medium", "Low"):
                priority = "Medium"
            improvements.append(
                {
                    "priority": priority,
                    "area": str(item.get("area", "General")).strip() or "General",
                    "suggestion": str(item.get("suggestion", "")).strip(),
                }
            )
        elif isinstance(item, str) and item.strip():
            improvements.append({"priority": "Medium", "area": "General", "suggestion": item.strip()})
    improvements = [i for i in improvements if i["suggestion"]]
    order = {"High": 0, "Medium": 1, "Low": 2}
    improvements.sort(key=lambda i: order[i["priority"]])

    rewrites = []
    for item in data.get("bullet_rewrites") or []:
        if isinstance(item, dict) and item.get("original") and item.get("improved"):
            rewrites.append({"original": str(item["original"]).strip(), "improved": str(item["improved"]).strip()})

    return {
        "ats_score": _clamp_score(data.get("ats_score")),
        "summary": str(data.get("summary", "")).strip(),
        "detected_role": str(data.get("detected_role", "")).strip(),
        "section_scores": {k: _clamp_score(scores_in.get(k)) for k in SCORE_SECTIONS},
        "matched_keywords": _str_list(data.get("matched_keywords")),
        "missing_keywords": _str_list(data.get("missing_keywords")),
        "strengths": _str_list(data.get("strengths")),
        "weaknesses": _str_list(data.get("weaknesses")),
        "improvements": improvements,
        "bullet_rewrites": rewrites,
        "missing_sections": _str_list(data.get("missing_sections")),
    }


def friendly_api_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "permission" in low or "401" in low or "403" in low:
        return "Gemini rejected the API key. Check that it is correct and has access to the model."
    if "429" in low or "quota" in low or "resource_exhausted" in low or "rate limit" in low:
        return "Gemini rate limit / quota reached. Wait a minute and try again."
    if "404" in low or "not found" in low:
        return "Model not found. Change the model name in the sidebar (e.g. gemini-3.5-flash)."
    if "timeout" in low or "unavailable" in low or "503" in low:
        return "Gemini is temporarily unavailable. Please try again shortly."
    return f"Gemini request failed: {msg[:300]}"


def analyze_resume(resume_text: str, job_description: str, api_key: str, model: str) -> dict:
    """Call Gemini and return a normalized result dict."""
    # Imported here so the rest of the app still loads if the SDK is missing.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.2,
    )
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(resume_text, job_description),
        config=config,
    )
    return normalize_result(parse_json_response(response.text))


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def score_band(score: int):
    if score >= 80:
        return "Excellent", "#16a34a"
    if score >= 65:
        return "Good", "#65a30d"
    if score >= 50:
        return "Needs work", "#d97706"
    return "Poor", "#dc2626"


def get_api_key(sidebar_value: str) -> str:
    if sidebar_value.strip():
        return sidebar_value.strip()
    try:
        secret = st.secrets.get("GEMINI_API_KEY", "")
        if secret:
            return str(secret)
    except Exception:
        pass  # no secrets.toml locally
    return os.environ.get("GEMINI_API_KEY", "")


def render_results(result: dict) -> None:
    score = result["ats_score"]
    label, color = score_band(score)

    st.markdown("---")
    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"""
            <div style="text-align:center;padding:1.2rem;border-radius:16px;
                        border:2px solid {color};">
              <div style="font-size:3.4rem;font-weight:800;color:{color};line-height:1">{score}</div>
              <div style="opacity:.7">out of 100</div>
              <div style="font-weight:700;color:{color};margin-top:.3rem">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if result["detected_role"]:
            st.markdown(f"**Target role:** {result['detected_role']}")
        st.write(result["summary"])

    st.subheader("Score breakdown")
    cols = st.columns(3)
    for i, (key, title) in enumerate(SCORE_SECTIONS.items()):
        val = result["section_scores"][key]
        with cols[i % 3]:
            st.metric(title, f"{val}/100")
            st.progress(val / 100)

    tab_fix, tab_kw, tab_sw, tab_rewrite = st.tabs(
        ["Improvements", "Keywords", "Strengths & weaknesses", "Bullet rewrites"]
    )

    with tab_fix:
        if result["missing_sections"]:
            st.warning("Missing sections: " + ", ".join(result["missing_sections"]))
        if not result["improvements"]:
            st.info("No specific improvements were returned.")
        icons = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
        for item in result["improvements"]:
            st.markdown(f"{icons[item['priority']]} **{item['priority']} · {item['area']}**")
            st.write(item["suggestion"])

    with tab_kw:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Matched keywords**")
            st.write(", ".join(result["matched_keywords"]) or "None found")
        with c2:
            st.markdown("**❌ Missing keywords**")
            st.write(", ".join(result["missing_keywords"]) or "None found")

    with tab_sw:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**💪 Strengths**")
            for s in result["strengths"] or ["None listed"]:
                st.markdown(f"- {s}")
        with c2:
            st.markdown("**⚠️ Weaknesses**")
            for w in result["weaknesses"] or ["None listed"]:
                st.markdown(f"- {w}")

    with tab_rewrite:
        if not result["bullet_rewrites"]:
            st.info("No bullet rewrites were returned.")
        for item in result["bullet_rewrites"]:
            st.markdown(f"**Before:** {item['original']}")
            st.markdown(f"**After:** {item['improved']}")
            st.divider()

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(result, indent=2),
        file_name="ats_report.json",
        mime="application/json",
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="AI Resume ATS Analyzer", page_icon="📄", layout="wide")
    st.title("📄 AI Resume ATS Analyzer")
    st.caption("Upload your resume, get an estimated ATS score and specific ways to improve it.")

    with st.sidebar:
        st.header("Settings")
        key_input = st.text_input(
            "Gemini API key",
            type="password",
            help="Get a free key at https://aistudio.google.com/apikey. "
            "Not needed if GEMINI_API_KEY is set in secrets or environment.",
        )
        model = st.text_input("Model", value=DEFAULT_MODEL)
        st.caption(
            "The score is an AI-based estimate, not the output of a real ATS. "
            "Use it as guidance. Your resume is sent to Google's Gemini API for analysis."
        )

    uploaded = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
    job_description = st.text_area(
        "Job description (optional, recommended for accurate keyword matching)",
        height=160,
        placeholder="Paste the job description here...",
    )

    if st.button("Analyze resume", type="primary", disabled=uploaded is None):
        api_key = get_api_key(key_input)
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            st.stop()
        if not model.strip():
            st.error("Please enter a model name.")
            st.stop()

        data = uploaded.getvalue()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_FILE_MB} MB.")
            st.stop()

        try:
            with st.spinner("Reading resume..."):
                resume_text = extract_resume_text(uploaded.name, data)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        try:
            with st.spinner("Analyzing with Gemini..."):
                result = analyze_resume(resume_text, job_description, api_key, model.strip())
        except ValueError as exc:  # bad JSON etc.
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(friendly_api_error(exc))
            st.stop()

        st.session_state["result"] = result

    if "result" in st.session_state:
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
