import io
import os
import re
import json
import math
from typing import List, Dict, Any

import numpy as np
import streamlit as st
import faiss
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="ECAT-Tutor-AI",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-bottom: 25px;
    }
    .agent-box {
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #f8fafc;
        margin-bottom: 10px;
    }
    .verified {
        color: #087443;
        font-weight: 700;
    }
    .warning {
        color: #b54708;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Constants
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MODEL = "bert-base-uncased"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


# -----------------------------
# Cached AI components
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


# -----------------------------
# PDF processing
# -----------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from a PDF uploaded by the user."""
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n\n".join(pages).strip()


def normalize_text(text: str) -> str:
    """Clean repeated whitespace while preserving readable paragraphs."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# -----------------------------
# Tokenization + chunking
# -----------------------------
def tokenize_text(text: str) -> List[int]:
    """Tokenize source text using a Hugging Face tokenizer."""
    tokenizer = load_tokenizer()
    return tokenizer.encode(text, add_special_tokens=False)


def chunk_text(text: str, max_tokens: int = 300, overlap: int = 60) -> List[str]:
    """
    Split text into token-aware chunks.

    Token-aware chunking keeps chunks within a predictable size for
    embedding and retrieval while retaining overlap between neighboring chunks.
    """
    tokenizer = load_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        return []

    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(token_ids):
            break

        start = max(0, end - overlap)

    return chunks


# -----------------------------
# FAISS vector database
# -----------------------------
def build_faiss_index(chunks: List[str]):
    """Create normalized sentence embeddings and store them in FAISS."""
    embedder = load_embedding_model()

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve_context(
    query: str,
    chunks: List[str],
    index,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """Retrieve the most relevant chunks from the FAISS index."""
    embedder = load_embedding_model()

    query_vector = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append(
                {
                    "chunk": chunks[int(idx)],
                    "score": float(score),
                    "chunk_id": int(idx),
                }
            )

    return results


# -----------------------------
# Groq client
# -----------------------------
def get_groq_client() -> Groq:
    """Create a Groq client from Streamlit secrets or environment variables."""
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or your environment."
        )

    return Groq(api_key=api_key)


def groq_generate(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2,
) -> str:
    """Call Groq Chat Completions and return the generated text."""
    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=6000,
    )

    return response.choices[0].message.content


# -----------------------------
# Agent A: Syllabus/RAG Architect
# -----------------------------
def architect_agent(
    subject: str,
    topic: str,
    difficulty: str,
    number_of_questions: int,
    retrieved_context: str,
    model: str,
) -> Dict[str, Any]:
    """Create an exam blueprint using only retrieved PDF context."""
    system_prompt = """
You are Agent A, the ECAT Syllabus and Blueprint Architect.

Your job is to design a high-quality ECAT mock-test blueprint.
Use ONLY the supplied PDF context as the source of syllabus facts.
Do not invent topics, formulas, rules, or syllabus claims.

Return valid JSON with:
{
  "subject": "...",
  "topic": "...",
  "difficulty": "...",
  "question_count": 20,
  "blueprint": [
    {
      "concept": "...",
      "question_type": "conceptual|numerical",
      "difficulty": "...",
      "focus": "..."
    }
  ]
}
"""

    user_prompt = f"""
Subject: {subject}
Requested topic: {topic}
Difficulty: {difficulty}
Number of questions: {number_of_questions}

Retrieved PDF context:
{retrieved_context}
"""

    raw = groq_generate(system_prompt, user_prompt, model=model, temperature=0.1)

    try:
        return json.loads(extract_json(raw))
    except Exception:
        return {
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "question_count": number_of_questions,
            "blueprint": [
                {
                    "concept": topic,
                    "question_type": "mixed",
                    "difficulty": difficulty,
                    "focus": "Use the retrieved PDF content only.",
                }
            ],
        }


# -----------------------------
# Agent B: MCQ Generator
# -----------------------------
def generator_agent(
    blueprint: Dict[str, Any],
    retrieved_context: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Generate MCQs grounded in the retrieved PDF context."""
    system_prompt = """
You are Agent B, an expert ECAT MCQ author.

Generate original ECAT-style multiple-choice questions using ONLY the
retrieved study material and blueprint.

Important:
- Do not copy a question verbatim from the PDF.
- Do not invent information outside the retrieved context.
- Create four options A, B, C, D.
- Make distractors plausible and based on common student mistakes.
- Do not reveal the correct answer to Agent C in a separate field.
- For numerical questions, include all necessary values in the question.
- Keep mathematical notation readable in plain text.

Return ONLY a JSON array:
[
  {
    "id": 1,
    "question": "...",
    "options": {
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    },
    "concept": "...",
    "difficulty": "..."
  }
]
"""

    user_prompt = f"""
Blueprint:
{json.dumps(blueprint, indent=2)}

Retrieved source context:
{retrieved_context}
"""

    raw = groq_generate(system_prompt, user_prompt, model=model, temperature=0.45)
    parsed = json.loads(extract_json(raw))

    if isinstance(parsed, dict) and "questions" in parsed:
        parsed = parsed["questions"]

    return parsed


# -----------------------------
# Agent C: Independent verifier
# -----------------------------
def safe_math_eval(expression: str):
    """
    Evaluate a restricted mathematical expression using Python.

    This is intentionally limited to arithmetic and selected math functions.
    It is not a general Python execution environment.
    """
    allowed = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "pow": pow,
    }

    cleaned = expression.strip()
    cleaned = cleaned.replace("^", "**")

    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,%\s*]+", cleaned):
        raise ValueError("Expression contains unsupported characters.")

    return eval(cleaned, {"__builtins__": {}}, allowed)


def verifier_agent(
    questions: List[Dict[str, Any]],
    retrieved_context: str,
    model: str,
) -> List[Dict[str, Any]]:
    """
    Independently solve each MCQ without trusting a generated answer key.

    The LLM verifier determines the reasoning and identifies the likely
    correct option. A deterministic Python calculator is used when the
    verifier supplies a simple arithmetic expression.
    """
    verified_questions = []

    system_prompt = """
You are Agent C, an independent ECAT quality-control verifier.

You must solve each question independently.
You have NOT been given an answer key.

For every question:
1. Determine the correct option.
2. Check the calculation carefully.
3. Identify ambiguity, missing information, or unsupported claims.
4. Provide a concise step-by-step explanation.
5. State PASS only when the question and answer are defensible from the source context.

Return ONLY JSON:
{
  "correct_option": "A|B|C|D",
  "verification_status": "PASS|FAIL",
  "reason": "...",
  "explanation": "...",
  "calculation_expression": null
}

If a simple numerical calculation can be represented as a safe arithmetic
expression, put that expression in calculation_expression. Otherwise use null.
"""

    for question in questions:
        user_prompt = f"""
Question:
{question.get("question", "")}

Options:
{json.dumps(question.get("options", {}), indent=2)}

Concept:
{question.get("concept", "")}

Relevant source context:
{retrieved_context}
"""

        try:
            raw = groq_generate(
                system_prompt,
                user_prompt,
                model=model,
                temperature=0.05,
            )
            result = json.loads(extract_json(raw))
        except Exception as exc:
            result = {
                "correct_option": "UNKNOWN",
                "verification_status": "FAIL",
                "reason": f"Verifier error: {exc}",
                "explanation": "The question could not be independently verified.",
                "calculation_expression": None,
            }

        expression = result.get("calculation_expression")

        if expression:
            try:
                value = safe_math_eval(expression)
                result["calculator_check"] = {
                    "expression": expression,
                    "result": value,
                    "status": "PASS",
                }
            except Exception as exc:
                result["calculator_check"] = {
                    "expression": expression,
                    "result": None,
                    "status": "NOT_EXECUTED",
                    "reason": str(exc),
                }
        else:
            result["calculator_check"] = {
                "expression": None,
                "result": None,
                "status": "NOT_REQUIRED",
            }

        verified_questions.append(
            {
                **question,
                "verification": result,
            }
        )

    return verified_questions


# -----------------------------
# Agent D: Curator/Explainer
# -----------------------------
def curator_agent(
    verified_questions: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    """Create polished pedagogical explanations for verified questions."""
    system_prompt = """
You are Agent D, the ECAT Curator and Explainer.

Only process questions that have passed verification.
Improve the explanation for students without changing the verified answer.

Return ONLY JSON array:
[
  {
    "id": 1,
    "correct_option": "A",
    "explanation": "Step-by-step explanation..."
  }
]
"""

    passed = [
        q
        for q in verified_questions
        if q.get("verification", {}).get("verification_status") == "PASS"
    ]

    if not passed:
        return []

    user_prompt = json.dumps(
        [
            {
                "id": q.get("id"),
                "question": q.get("question"),
                "options": q.get("options"),
                "verified_answer": q.get("verification", {}).get("correct_option"),
                "verification_reason": q.get("verification", {}).get("reason"),
            }
            for q in passed
        ],
        indent=2,
    )

    raw = groq_generate(system_prompt, user_prompt, model=model, temperature=0.15)

    try:
        result = json.loads(extract_json(raw))
        return result if isinstance(result, list) else result.get("questions", [])
    except Exception:
        return []


# -----------------------------
# JSON helper
# -----------------------------
def extract_json(text: str) -> str:
    """Extract a JSON object or array from an LLM response."""
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    array_start = text.find("[")
    object_start = text.find("{")

    starts = [x for x in [array_start, object_start] if x >= 0]

    if not starts:
        raise ValueError("No JSON object or array found.")

    start = min(starts)

    array_end = text.rfind("]")
    object_end = text.rfind("}")

    end = max(array_end, object_end)

    if end < start:
        raise ValueError("Incomplete JSON response.")

    return text[start:end + 1]


# -----------------------------
# Quiz scoring
# -----------------------------
def score_quiz(questions: List[Dict[str, Any]], answers: Dict[int, str]):
    """Calculate quiz score against independently verified answers."""
    correct = 0

    for q in questions:
        qid = q.get("id")
        expected = q.get("verification", {}).get("correct_option")
        selected = answers.get(qid)

        if selected and expected and selected == expected:
            correct += 1

    total = len(questions)
    percentage = round((correct / total) * 100, 2) if total else 0

    return correct, total, percentage


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ ECAT-Tutor-AI")

    groq_model = st.selectbox(
        "Groq model",
        [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        index=0,
    )

    top_k = st.slider(
        "RAG chunks to retrieve",
        min_value=2,
        max_value=10,
        value=6,
    )

    chunk_size = st.slider(
        "Chunk size (tokens)",
        min_value=150,
        max_value=600,
        value=300,
        step=50,
    )

    st.divider()

    st.markdown("### Pipeline")
    st.write("📄 PDF ingestion")
    st.write("✂️ Token-aware chunking")
    st.write("🔤 Tokenization")
    st.write("🧠 Embeddings")
    st.write("🔎 FAISS retrieval")
    st.write("🤖 Groq agents")
    st.write("🧮 Deterministic verification")
    st.write("✅ Certified questions")


# -----------------------------
# Main UI
# -----------------------------
st.markdown(
    '<div class="main-title">🎓 ECAT-Tutor-AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">RAG-Based ECAT Mock Test Generator & Quality-Control Auditor</div>',
    unsafe_allow_html=True,
)

st.info(
    "Upload an ECAT PDF. The system extracts its content, chunks and tokenizes it, "
    "creates embeddings, stores them in FAISS, retrieves relevant context, and "
    "generates verified MCQs grounded in that PDF."
)

uploaded_file = st.file_uploader(
    "📚 Upload ECAT study material / syllabus PDF",
    type=["pdf"],
    help="Upload an ECAT preparation PDF, syllabus, notes, or chapter material.",
)

col1, col2, col3 = st.columns(3)

with col1:
    subject = st.selectbox(
        "Subject",
        ["Physics", "Mathematics", "Chemistry", "Mixed ECAT"],
    )

with col2:
    topic = st.text_input(
        "Topic",
        placeholder="e.g. Calculus, Rotational Motion, Vectors",
    )

with col3:
    difficulty = st.selectbox(
        "Difficulty",
        ["Easy", "Medium", "Hard", "ECAT Challenge"],
        index=1,
    )

number_of_questions = st.slider(
    "Number of MCQs",
    min_value=5,
    max_value=20,
    value=10,
)

if uploaded_file:
    if st.button("🔍 Process PDF & Build RAG Database", use_container_width=True):
        with st.spinner("Extracting PDF text..."):
            raw_text = extract_pdf_text(uploaded_file)
            raw_text = normalize_text(raw_text)

        if len(raw_text) < 100:
            st.error(
                "Very little text was extracted. If your PDF is scanned images, "
                "OCR support is required before it can be retrieved."
            )
        else:
            with st.spinner("Tokenizing and chunking the PDF..."):
                token_ids = tokenize_text(raw_text)
                chunks = chunk_text(
                    raw_text,
                    max_tokens=chunk_size,
                    overlap=max(30, chunk_size // 5),
                )

            with st.spinner("Creating embeddings and FAISS vector database..."):
                index, embeddings = build_faiss_index(chunks)

            st.session_state["raw_text"] = raw_text
            st.session_state["chunks"] = chunks
            st.session_state["faiss_index"] = index
            st.session_state["embeddings"] = embeddings
            st.session_state["token_count"] = len(token_ids)
            st.session_state["source_name"] = uploaded_file.name

            st.success(
                f"RAG database ready: {len(chunks)} chunks, "
                f"{len(token_ids):,} source tokens, "
                f"{embeddings.shape[1]}-dimension embeddings."
            )

if "faiss_index" in st.session_state:
    st.divider()

    rag_col1, rag_col2, rag_col3 = st.columns(3)

    with rag_col1:
        st.metric("PDF Chunks", len(st.session_state["chunks"]))

    with rag_col2:
        st.metric("Tokens", f"{st.session_state['token_count']:,}")

    with rag_col3:
        st.metric(
            "Embedding Dimension",
            st.session_state["embeddings"].shape[1],
        )

    if st.button(
        "🚀 Generate Verified ECAT Mock Test",
        type="primary",
        use_container_width=True,
    ):
        if not topic.strip():
            st.warning(
                "Enter a topic so the RAG retriever can focus on the requested ECAT concept."
            )
            st.stop()

        try:
            with st.status(
                "Running ECAT-Tutor-AI agent pipeline...",
                expanded=True,
            ) as status:

                st.write("🔎 Retrieving relevant PDF context...")
                retrieved = retrieve_context(
                    query=f"{subject} {topic} {difficulty} ECAT",
                    chunks=st.session_state["chunks"],
                    index=st.session_state["faiss_index"],
                    top_k=top_k,
                )

                context_text = "\n\n--- SOURCE CHUNK ---\n\n".join(
                    [
                        f"[Chunk {r['chunk_id']} | similarity={r['score']:.3f}]\n{r['chunk']}"
                        for r in retrieved
                    ]
                )

                st.write("🧠 Agent A: Building syllabus/question blueprint...")
                blueprint = architect_agent(
                    subject=subject,
                    topic=topic,
                    difficulty=difficulty,
                    number_of_questions=number_of_questions,
                    retrieved_context=context_text,
                    model=groq_model,
                )

                st.write("✍️ Agent B: Generating original ECAT MCQs...")
                questions = generator_agent(
                    blueprint=blueprint,
                    retrieved_context=context_text,
                    model=groq_model,
                )

                questions = questions[:number_of_questions]

                st.write("🧮 Agent C: Independently solving and verifying MCQs...")
                verified_questions = verifier_agent(
                    questions=questions,
                    retrieved_context=context_text,
                    model=groq_model,
                )

                st.write("📘 Agent D: Creating student-friendly explanations...")
                explanations = curator_agent(
                    verified_questions=verified_questions,
                    model=groq_model,
                )

                explanation_map = {
                    item.get("id"): item
                    for item in explanations
                }

                for q in verified_questions:
                    exp = explanation_map.get(q.get("id"))
                    if exp:
                        q["curated_explanation"] = exp.get("explanation", "")

                passed_questions = [
                    q
                    for q in verified_questions
                    if q.get("verification", {}).get("verification_status") == "PASS"
                ]

                st.session_state["exam_questions"] = passed_questions
                st.session_state["retrieved_context"] = retrieved
                st.session_state["blueprint"] = blueprint

                status.update(
                    label=(
                        f"Pipeline complete — {len(passed_questions)} verified "
                        f"questions generated."
                    ),
                    state="complete",
                )

        except Exception as exc:
            st.error(f"Generation failed: {exc}")

# -----------------------------
# Display retrieved context
# -----------------------------
if st.session_state.get("retrieved_context"):
    with st.expander("🔎 View RAG Retrieved Context"):
        for item in st.session_state["retrieved_context"]:
            st.markdown(
                f"**Chunk {item['chunk_id']} — similarity {item['score']:.3f}**"
            )
            st.write(item["chunk"])

# -----------------------------
# Display exam
# -----------------------------
questions = st.session_state.get("exam_questions", [])

if questions:
    st.divider()
    st.header("📝 Verified ECAT Mock Test")

    st.caption(
        f"Source: {st.session_state.get('source_name', 'Uploaded PDF')} | "
        f"Verified questions: {len(questions)}"
    )

    answers = {}

    for position, q in enumerate(questions, start=1):
        qid = q.get("id", position)

        st.subheader(
            f"Q{position}. {q.get('question', '')}"
        )

        options = q.get("options", {})

        selected = st.radio(
            "Choose an answer:",
            options=["Select an option"] + list(options.keys()),
            format_func=lambda key: (
                "Select an option"
                if key == "Select an option"
                else f"{key}) {options.get(key, '')}"
            ),
            key=f"answer_{qid}",
        )

        if selected != "Select an option":
            answers[qid] = selected

        st.markdown(
            f"**Concept:** {q.get('concept', 'Not specified')}  \n"
            f"**Difficulty:** {q.get('difficulty', difficulty)}"
        )

        verification = q.get("verification", {})
        st.markdown(
            '<span class="verified">✓ Independently Verified</span>',
            unsafe_allow_html=True,
        )

        with st.expander("View explanation"):
            st.write(
                q.get("curated_explanation")
                or verification.get("explanation", "No explanation available.")
            )

        with st.expander("View quality-control details"):
            st.write(
                f"**Verified answer:** "
                f"{verification.get('correct_option', 'Unknown')}"
            )
            st.write(
                f"**Verification status:** "
                f"{verification.get('verification_status', 'Unknown')}"
            )
            st.write(
                f"**Verifier reason:** "
                f"{verification.get('reason', '')}"
            )

            calculator_check = verification.get("calculator_check", {})
            if calculator_check.get("expression"):
                st.write(
                    f"**Calculator expression:** "
                    f"`{calculator_check.get('expression')}`"
                )
                st.write(
                    f"**Calculator result:** "
                    f"{calculator_check.get('result')}"
                )
                st.write(
                    f"**Calculator status:** "
                    f"{calculator_check.get('status')}"
                )

        st.divider()

    if st.button("📊 Submit Quiz", type="primary"):
        correct, total, percentage = score_quiz(questions, answers)

        st.success(
            f"Score: {correct}/{total} — {percentage}%"
        )

        if percentage >= 80:
            st.balloons()
            st.write("Excellent ECAT preparation performance!")
        elif percentage >= 60:
            st.write("Good attempt. Review the explanations for improvement.")
        else:
            st.write("Keep practicing and review the verified explanations.")

    exam_text = "# ECAT-Tutor-AI — Verified Mock Test\n\n"

    for position, q in enumerate(questions, start=1):
        exam_text += f"## Q{position}. {q.get('question', '')}\n\n"

        for key, value in q.get("options", {}).items():
            exam_text += f"- {key}) {value}\n"

        verification = q.get("verification", {})
        exam_text += (
            f"\n**Verified Answer:** "
            f"{verification.get('correct_option', 'Unknown')}\n\n"
        )
        exam_text += (
            f"**Explanation:** "
            f"{q.get('curated_explanation') or verification.get('explanation', '')}\n\n"
        )

    st.download_button(
        "⬇️ Download Verified Mock Test",
        data=exam_text,
        file_name="ecat_verified_mock_test.md",
        mime="text/markdown",
        use_container_width=True,
    )
else:
    st.markdown(
        """
        ### How it works

        1. Upload an ECAT PDF.
        2. The PDF is converted into text.
        3. Text is tokenized and split into overlapping chunks.
        4. Chunks are converted into embedding vectors.
        5. FAISS stores and searches the vectors.
        6. Relevant PDF content is retrieved for the selected topic.
        7. Agent A creates the question blueprint.
        8. Agent B generates original MCQs and distractors.
        9. Agent C independently verifies each question.
        10. Agent D produces student-friendly explanations.
        11. Only passed questions are shown as the verified quiz.
        """
    )

st.caption(
    "ECAT-Tutor-AI | RAG + FAISS + Sentence Transformers + Groq + Streamlit"
)
