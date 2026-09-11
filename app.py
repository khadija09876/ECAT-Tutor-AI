import os
import json
import re
from typing import Any, Dict, List

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


st.set_page_config(
    page_title="ECAT-Tutor-AI",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        color: #777;
        margin-bottom: 28px;
    }
    .result-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎓 ECAT-Tutor-AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload your ECAT book and let AI create summaries, notes, MCQs, and quizzes.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def get_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or as an environment variable."
        )
    return Groq(api_key=api_key)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_pages(uploaded_file) -> List[Dict[str, Any]]:
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append({"page": page_number, "text": text})

    return pages


def create_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 80,
) -> List[Dict[str, Any]]:
    tokenizer = get_tokenizer()
    chunks = []

    for page in pages:
        token_ids = tokenizer.encode(
            page["text"],
            add_special_tokens=False,
        )

        start = 0

        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))

            chunk_text = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
            ).strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                    }
                )

            if end >= len(token_ids):
                break

            start = max(end - overlap, start + 1)

    return chunks


def create_vector_index(chunks: List[Dict[str, Any]]):
    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_relevant_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    index,
    top_k: int = 6,
):
    model = get_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, ids = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append(
                {
                    "text": chunks[int(idx)]["text"],
                    "page": chunks[int(idx)]["page"],
                    "score": float(score),
                }
            )

    return results


def extract_json(text: str):
    text = text.strip()

    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("AI returned invalid JSON.")


def ask_groq(prompt: str):
    client = get_groq_client()

    response = client.chat.completions.create(
       model="openai/gpt-oss-120b",
        temperature=0.15,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are ECAT-Tutor-AI. "
                    "Create accurate educational material using only the supplied "
                    "uploaded-book content. Do not invent source-specific facts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return extract_json(response.choices[0].message.content)


def context_text(context: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[Book Page {item['page']}] {item['text']}"
        for item in context
    )


def generate_summary(context: List[Dict[str, Any]], topic: str):
    prompt = f"""
Create a concise ECAT study summary from the supplied book content.

Requested topic:
{topic or "Use the main concepts available in the uploaded book."}

Return JSON:
{{
  "title": "string",
  "summary": "clear exam-focused summary",
  "key_points": ["important point 1", "important point 2"],
  "formulas": ["formula 1", "formula 2"],
  "quick_revision": ["revision point 1", "revision point 2"]
}}

Only use the supplied book content.

BOOK CONTENT:
{context_text(context)}
"""
    return ask_groq(prompt)


def generate_notes(context: List[Dict[str, Any]], topic: str):
    prompt = f"""
Create detailed but easy-to-revise ECAT notes from the supplied book content.

Requested topic:
{topic or "Use the retrieved book material."}

Return JSON:
{{
  "title": "string",
  "concepts": ["concept with concise explanation"],
  "definitions": ["important definition"],
  "formulas": ["important formula and meaning"],
  "important_points": ["exam-relevant point"],
  "exam_tips": ["useful exam tip"]
}}

Use only the supplied book content.

BOOK CONTENT:
{context_text(context)}
"""
    return ask_groq(prompt)


def generate_mcqs(
    context: List[Dict[str, Any]],
    topic: str,
    difficulty: str,
    count: int,
):
    prompt = f"""
Create exactly {count} ECAT-style MCQs from the supplied book content.

Topic:
{topic or "Relevant material from the uploaded book"}

Difficulty:
{difficulty}

Rules:
- Use only the supplied book content.
- Exactly four options: A, B, C, D.
- Exactly one correct answer.
- Include conceptual and numerical questions when supported by the source.
- Numerical answers must be calculated carefully.
- Provide a short explanation.
- Questions must be suitable for ECAT preparation.

Return JSON:
{{
  "questions": [
    {{
      "question": "question text",
      "A": "option A",
      "B": "option B",
      "C": "option C",
      "D": "option D",
      "answer": "A",
      "explanation": "clear explanation"
    }}
  ]
}}

BOOK CONTENT:
{context_text(context)}
"""
    data = ask_groq(prompt)
    return data.get("questions", [])


def generate_quiz(context: List[Dict[str, Any]], topic: str, difficulty: str):
    prompt = f"""
Create exactly 10 ECAT practice quiz questions from the supplied book content.

Topic:
{topic or "Relevant material from the uploaded book"}

Difficulty:
{difficulty}

Rules:
- Use only the supplied book content.
- Four options A, B, C, D.
- Exactly one correct answer.
- Mix conceptual and numerical questions when possible.
- Verify numerical calculations.
- Do not reveal the answer in the question.

Return JSON:
{{
  "questions": [
    {{
      "question": "question text",
      "A": "option A",
      "B": "option B",
      "C": "option C",
      "D": "option D",
      "answer": "A",
      "explanation": "explanation shown after submission"
    }}
  ]
}}

BOOK CONTENT:
{context_text(context)}
"""
    data = ask_groq(prompt)
    return data.get("questions", [])


def render_questions(questions: List[Dict[str, Any]], quiz_mode: bool):
    if not questions:
        st.warning("No questions were generated. Try another topic.")
        return

    if quiz_mode:
        st.markdown("### 🎯 ECAT Quiz")

        with st.form("ecat_quiz_form"):
            answers = {}

            for i, question in enumerate(questions):
                st.markdown(f"#### Q{i + 1}. {question.get('question', '')}")

                answers[i] = st.radio(
                    "Select your answer",
                    ["A", "B", "C", "D"],
                    format_func=lambda x, q=question: f"{x}. {q.get(x, '')}",
                    key=f"quiz_answer_{i}",
                )

            submitted = st.form_submit_button(
                "Submit Quiz",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            score = 0

            for i, question in enumerate(questions):
                correct = str(question.get("answer", "")).upper()

                if answers.get(i) == correct:
                    score += 1

            percentage = round((score / len(questions)) * 100)

            st.divider()
            st.subheader("📊 Quiz Result")
            st.metric(
                "Score",
                f"{score} / {len(questions)}",
            )
            st.progress(percentage / 100)
            st.write(f"Accuracy: **{percentage}%**")

            if percentage >= 80:
                st.success("Excellent performance! Keep practicing.")
            elif percentage >= 60:
                st.info("Good attempt. Revise the weaker concepts.")
            else:
                st.warning("Revise the topic and try the quiz again.")

            st.markdown("### Answer Review")

            for i, question in enumerate(questions):
                correct = str(question.get("answer", "")).upper()
                selected = answers.get(i)

                if selected == correct:
                    st.success(f"Q{i + 1}: Correct")
                else:
                    st.error(
                        f"Q{i + 1}: Your answer: {selected} | "
                        f"Correct answer: {correct}"
                    )

                st.caption(question.get("explanation", ""))

    else:
        st.markdown("### 🧠 Generated MCQs")

        for i, question in enumerate(questions):
            st.markdown(
                f"#### Q{i + 1}. {question.get('question', '')}"
            )

            st.markdown(f"**A.** {question.get('A', '')}")
            st.markdown(f"**B.** {question.get('B', '')}")
            st.markdown(f"**C.** {question.get('C', '')}")
            st.markdown(f"**D.** {question.get('D', '')}")

            with st.expander("Show answer & explanation"):
                st.success(
                    f"Correct Answer: {question.get('answer', '')}"
                )
                st.write(question.get("explanation", ""))


# -----------------------------
# Session state
# -----------------------------
if "book_ready" not in st.session_state:
    st.session_state.book_ready = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if "summary" not in st.session_state:
    st.session_state.summary = None

if "notes" not in st.session_state:
    st.session_state.notes = None

if "mcqs" not in st.session_state:
    st.session_state.mcqs = []

if "quiz" not in st.session_state:
    st.session_state.quiz = []


# -----------------------------
# Main user interface
# -----------------------------
st.markdown("## 📚 Upload Your ECAT Book")

uploaded_file = st.file_uploader(
    "Choose your ECAT book in PDF format",
    type=["pdf"],
    help="Upload a text-based PDF. Scanned image-only PDFs require OCR.",
)

if uploaded_file:
    if st.button(
        "🚀 Process My Book",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Preparing your book..."):
                pages = extract_pdf_pages(uploaded_file)

                if not pages:
                    st.error(
                        "No selectable text was found in this PDF. "
                        "Please use a text-based PDF or add OCR support."
                    )
                    st.stop()

                chunks = create_chunks(pages)
                index = create_vector_index(chunks)

                st.session_state.book_ready = True
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.source_name = uploaded_file.name

                st.session_state.summary = None
                st.session_state.notes = None
                st.session_state.mcqs = []
                st.session_state.quiz = []

            st.success("Your ECAT book is ready! 🎉")

        except Exception as exc:
            st.error(f"Book processing failed: {exc}")


if st.session_state.book_ready:
    st.divider()

    st.success(
        f"📖 Book loaded: {st.session_state.source_name}"
    )

    col1, col2 = st.columns(2)

    with col1:
        subject = st.selectbox(
            "Subject",
            ["Physics", "Mathematics", "Chemistry", "Mixed ECAT"],
        )

    with col2:
        difficulty = st.selectbox(
            "MCQ / Quiz Difficulty",
            ["Easy", "Medium", "Hard", "ECAT Challenge"],
            index=1,
        )

    topic = st.text_input(
        "Chapter or topic (optional)",
        placeholder="Example: Rotational Motion",
    )

    st.markdown("### What do you want to generate?")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        summary_button = st.button(
            "📝 Summary",
            use_container_width=True,
        )

    with c2:
        notes_button = st.button(
            "📌 Study Notes",
            use_container_width=True,
        )

    with c3:
        mcq_button = st.button(
            "🧠 MCQs",
            use_container_width=True,
        )

    with c4:
        quiz_button = st.button(
            "🎯 Start Quiz",
            use_container_width=True,
        )

    query = f"{subject}. {topic}" if topic else subject

    if summary_button:
        try:
            with st.spinner("Creating your summary..."):
                context = retrieve_relevant_chunks(
                    query,
                    st.session_state.chunks,
                    st.session_state.index,
                )

                st.session_state.summary = generate_summary(
                    context,
                    topic,
                )

            st.success("Summary generated.")

        except Exception as exc:
            st.error(f"Summary generation failed: {exc}")

    if notes_button:
        try:
            with st.spinner("Creating your study notes..."):
                context = retrieve_relevant_chunks(
                    query,
                    st.session_state.chunks,
                    st.session_state.index,
                )

                st.session_state.notes = generate_notes(
                    context,
                    topic,
                )

            st.success("Study notes generated.")

        except Exception as exc:
            st.error(f"Notes generation failed: {exc}")

    if mcq_button:
        try:
            with st.spinner("Creating ECAT MCQs..."):
                context = retrieve_relevant_chunks(
                    query,
                    st.session_state.chunks,
                    st.session_state.index,
                )

                st.session_state.mcqs = generate_mcqs(
                    context,
                    topic,
                    difficulty,
                    10,
                )

            st.success("MCQs generated.")

        except Exception as exc:
            st.error(f"MCQ generation failed: {exc}")

    if quiz_button:
        try:
            with st.spinner("Preparing your ECAT quiz..."):
                context = retrieve_relevant_chunks(
                    query,
                    st.session_state.chunks,
                    st.session_state.index,
                )

                st.session_state.quiz = generate_quiz(
                    context,
                    topic,
                    difficulty,
                )

            st.success("Quiz is ready.")

        except Exception as exc:
            st.error(f"Quiz generation failed: {exc}")


# -----------------------------
# Generated content
# -----------------------------
if st.session_state.summary:
    st.divider()
    st.subheader("📝 AI Summary")

    summary = st.session_state.summary

    st.markdown(f"### {summary.get('title', 'ECAT Summary')}")
    st.write(summary.get("summary", ""))

    with st.expander("📌 Key Points", expanded=True):
        for item in summary.get("key_points", []):
            st.markdown(f"- {item}")

    with st.expander("📐 Important Formulas"):
        for item in summary.get("formulas", []):
            st.markdown(f"- {item}")

    with st.expander("⚡ Quick Revision"):
        for item in summary.get("quick_revision", []):
            st.markdown(f"- {item}")


if st.session_state.notes:
    st.divider()
    st.subheader("📚 Study Notes")

    notes = st.session_state.notes

    st.markdown(f"### {notes.get('title', 'ECAT Study Notes')}")

    with st.expander("Concepts", expanded=True):
        for item in notes.get("concepts", []):
            st.markdown(f"- {item}")

    with st.expander("Definitions"):
        for item in notes.get("definitions", []):
            st.markdown(f"- {item}")

    with st.expander("Formulas"):
        for item in notes.get("formulas", []):
            st.markdown(f"- {item}")

    with st.expander("Important Points"):
        for item in notes.get("important_points", []):
            st.markdown(f"- {item}")

    with st.expander("Exam Tips"):
        for item in notes.get("exam_tips", []):
            st.markdown(f"- {item}")


if st.session_state.mcqs:
    st.divider()
    render_questions(
        st.session_state.mcqs,
        quiz_mode=False,
    )


if st.session_state.quiz:
    st.divider()
    render_questions(
        st.session_state.quiz,
        quiz_mode=True,
    )
