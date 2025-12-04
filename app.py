from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io

# -------------------------------------
# FastAPI Setup
# -------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/text")
async def process_text(prompt: str = Form(...), context: str = Form("")):
    response = generate_gemini_response(prompt, context)
    return {"response": response}

@app.post("/image")
async def process_image(file: UploadFile = File(...)):
    """
    Returns OCR text + estimated confidence using Gemini.
    """
    try:
        image_bytes = await file.read()
        text_with_conf = extract_text_from_image(io.BytesIO(image_bytes))
        return {"text": text_with_conf}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/pdf")
async def process_pdf(file: UploadFile = File(...)):
    """
    Returns PDF text + estimated confidence using Gemini+PyPDF2+pytesseract.
    """
    try:
        pdf_bytes = await file.read()
        text_with_conf = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        return {"text": text_with_conf}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Returns transcript + structured summary for audio.
    """
    try:
        audio_bytes = await file.read()
        text_and_summary = extract_text_from_audio(io.BytesIO(audio_bytes))
        return {"text": text_and_summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/youtube")
async def process_youtube(url: str = Form(...)):
    text = extract_transcript_details(url)
    return {"text": text}



# -------------------------------------
# STREAMLIT + AGENT CODE
# -------------------------------------
import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from PIL import Image
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
import speech_recognition as sr
from pydub import AudioSegment

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------
# AGENT SYSTEM PROMPT (Structured Output + Intent)
# -------------------------------------
AGENT_SYSTEM_PROMPT = """
You are an advanced multimodal AI agent. Your job is to:
- Understand the user's goal
- Use extracted content as context
- Choose the correct task
- Follow STRICT output formats

You MUST determine the user’s intent from:
- Their message
- Extracted OCR/PDF/Audio/YouTube content
- Conversation history (if described)

============================================
INTENT DETECTION TASKS (CHOOSE EXACTLY ONE)
============================================

1. SUMMARIZATION
Output MUST be exactly:
[ONE_LINE_SUMMARY]
<single concise sentence>

[THREE_BULLETS]
- bullet 1
- bullet 2
- bullet 3

[FIVE_SENTENCE_SUMMARY]
Sentence 1.
Sentence 2.
Sentence 3.
Sentence 4.
Sentence 5.

2. SENTIMENT_ANALYSIS
Output MUST be exactly:
[LABEL]
Positive / Neutral / Negative

[CONFIDENCE]
0.x (float between 0 and 1)

[JUSTIFICATION]
One short sentence explaining why.

3. CODE_EXPLANATION
Output MUST be exactly:
[CODE_EXPLANATION]
Explain what the code does.

[BUGS]
Mention any bugs or potential issues. If none, say "No obvious bugs."

[TIME_COMPLEXITY]
Give Big-O time complexity.

[IMPROVEMENTS]
Give 1–3 concrete improvements.

4. QUESTION_ANSWERING
If user asks a question about the content, answer directly and clearly:
[ANSWER]
<your answer>

5. AUDIO_SUMMARY or YOUTUBE_SUMMARY
When summarizing transcripts, use the SAME SUMMARIZATION FORMAT as above.

============================================
FOLLOW-UP QUESTION RULE (VERY IMPORTANT)
============================================
If and only if:
- User intent is unclear, OR
- Multiple tasks are equally plausible,

Then you MUST respond ONLY with:
[CLARIFICATION_QUESTION]
<one short clarifying question>

Do NOT perform any other task in that case.

============================================
GENERAL RESPONSE RULES
============================================
- Always follow the correct format blocks exactly.
- Never mix multiple task formats in one answer.
- Never hallucinate content not implied by the context.
- Prefer concise, clear language.
"""

# -------------------------------------
# GEMINI RESPONSE GENERATOR
# -------------------------------------
def generate_gemini_response(prompt, context: str = ""):
    """
    Central LLM call. Enforces structured output via prompt.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        full_prompt = f"""
{AGENT_SYSTEM_PROMPT}

[EXTRACTED_CONTENT]
{context}

[USER_MESSAGE]
{prompt}

Your job:
1. Infer the correct task.
2. Follow the exact output format for that task.
3. If intent is unclear, respond ONLY with [CLARIFICATION_QUESTION] block.
"""

        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return f"❌ AI Error: {str(e)}"


# -------------------------------------
# YOUTUBE HELPERS
# -------------------------------------
def extract_video_id(url: str):
    patterns = [
        r"v=([^&]+)", r"youtu\.be/([^?]+)",
        r"shorts/([^?]+)", r"embed/([^?]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def extract_transcript_details(url: str) -> str:
    """
    Fetches the raw transcript text for a YouTube URL using youtube-transcript-api.
    This text is then used as context for the Gemini agent.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return "❌ Could not extract video ID from the URL. Please check the URL."

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([item["text"] for item in transcript])
        return text
    except TranscriptsDisabled:
        return "❌ Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "❌ No transcript found for this video."
    except Exception as e:
        return f"❌ Error while fetching transcript: {str(e)}"


# -------------------------------------
# OCR IMAGE WITH GEMINI + ESTIMATED CONFIDENCE
# -------------------------------------
def extract_text_from_image(image_file):
    """
    Uses Gemini for OCR text + simple heuristic confidence estimate.
    Returns a human-readable string: TEXT + CONFIDENCE.
    """
    try:
        # image_file may be BytesIO (FastAPI) or UploadedFile (Streamlit)
        if hasattr(image_file, "read"):
            image_bytes = image_file.read()
        else:
            # If already bytes-like
            image_bytes = image_file

        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content([
            {"mime_type": "image/png", "data": image_bytes},
            "Extract all readable text from this image. Return only the text, no commentary."
        ])
        extracted = resp.text.strip() if resp.text else ""

        # Gemini doesn't expose numeric OCR confidence, so we approximate:
        # longer, denser text => higher confidence.
        if not extracted:
            confidence = 0.0
        else:
            # crude heuristic: normalized by 500 characters
            confidence = len(extracted) / 500.0
            if confidence > 1.0:
                confidence = 1.0

        return f"TEXT:\n{extracted}\n\nESTIMATED_CONFIDENCE: {round(confidence, 2)}"

    except Exception as e:
        return f"❌ Image OCR Error: {str(e)}"


# -------------------------------------
# PDF TEXT + CONFIDENCE (PyPDF2 + pytesseract)
# -------------------------------------
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from PDF; if no text, falls back to OCR.
    Adds a heuristic confidence score based on text amount.
    """
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        # Fallback to OCR on page images if no or very little text
        if not text.strip():
            pdf_file.seek(0)
            imgs = convert_from_bytes(pdf_file.read())
            for img in imgs:
                text += pytesseract.image_to_string(img)

        cleaned = text.strip()

        # Heuristic: more content => more confidence
        if not cleaned:
            confidence = 0.0
        else:
            confidence = len(cleaned) / 2000.0
            if confidence > 1.0:
                confidence = 1.0

        return f"TEXT:\n{cleaned}\n\nESTIMATED_CONFIDENCE: {round(confidence, 2)}"

    except Exception as e:
        return f"❌ PDF OCR Error: {str(e)}"


# -------------------------------------
# AUDIO → TEXT → SUMMARY PIPELINE
# -------------------------------------
def extract_text_from_audio(audio_file):
    """
    1. Converts audio to WAV
    2. Transcribes with SpeechRecognition (Google)
    3. Summarizes using Gemini with enforced 3-part format
    Returns TRANSCRIPT + SUMMARY.
    """
    try:
        recog = sr.Recognizer()
        if hasattr(audio_file, "read"):
            audio_bytes = audio_file.read()
        else:
            audio_bytes = audio_file

        # Convert to WAV
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Transcribe
        with sr.AudioFile(wav_io) as source:
            audio_data = recog.record(source)
            transcript = recog.recognize_google(audio_data)

        # Summarize using existing structured prompt
        summary_prompt = f"Please summarize the following audio transcript.\n\n{transcript}"
        summary = generate_gemini_response(summary_prompt, context=transcript)

        return f"TRANSCRIPT:\n{transcript}\n\nSUMMARY:\n{summary}"

    except Exception as e:
        return f"❌ Audio Error: {str(e)}"


# -------------------------------------
# STREAMLIT UI
# -------------------------------------
st.set_page_config(page_title="AI Multi-Input Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Multi-Input Agent")
st.caption("Text • Image OCR • PDF • Audio • YouTube | Intent-Aware Agent")

with st.sidebar:
    input_type = st.radio("Select Input Type:", ["Text", "Image", "PDF", "Audio", "YouTube"])
    uploaded_file = None
    youtube_url = None

    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    elif input_type == "PDF":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    elif input_type == "Audio":
        uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])

    elif input_type == "YouTube":
        youtube_url = st.text_input("Enter YouTube URL")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "extracted" in msg:
            st.expander("Extracted Content").write(msg["extracted"])

# -------------------------------------
# ChatGPT-like: auto extraction + always show chat box
# -------------------------------------
extracted_text = ""
user_input = ""

# Auto extraction when a file is uploaded
if uploaded_file:
    if input_type == "Image":
        extracted_text = extract_text_from_image(uploaded_file)
    elif input_type == "PDF":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif input_type == "Audio":
        extracted_text = extract_text_from_audio(uploaded_file)

    st.session_state.messages.append({
        "role": "assistant",
        "content": "✔ File processed! Ask anything you want about the content.",
        "extracted": extracted_text
    })

elif youtube_url:
    extracted_text = extract_transcript_details(youtube_url)
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🎥 YouTube transcript ready! Ask anything.",
        "extracted": extracted_text
    })

# Always show chat input
user_input = st.chat_input("Ask anything (summary, sentiment, explanation, etc.)...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Use most recent extracted content as context
    last_extracted = ""
    for m in reversed(st.session_state.messages):
        if "extracted" in m:
            last_extracted = m["extracted"]
            break

    with st.spinner("🤔 Thinking..."):
        ai_response = generate_gemini_response(user_input, last_extracted)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })

    st.rerun()
