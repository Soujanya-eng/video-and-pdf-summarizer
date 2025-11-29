import os
import fitz # PyMuPDF
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import traceback 
import warnings 

# --- Extractive Summarization Dependencies (Efficient & Stable) ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
from nltk.tokenize import sent_tokenize

# --- Multimodal Dependencies ---
import whisper 
# Note: MoviePy/OpenCV are used implicitly by Whisper/Flask for file handling.

# Suppress warnings that clutter the console (often from whisper/moviepy)
warnings.filterwarnings("ignore")

# --- Configuration (MULTIMODAL) ---
UPLOAD_FOLDER = 'uploads'
# Allowed extensions now include common audio/video types
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'mp4', 'avi', 'mov', 'mp3', 'wav'} 
LANGUAGE = "english"
SENTENCES_COUNT_RATIO = 0.25 # Target summary length (25% of sentences)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'multimodal_summarizer_key'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # Increased limit to 100MB for video/audio

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
    nltk.download('stopwords')

# --- Global Summarizer Setup (Efficient TextRank) ---
try:
    stemmer = Stemmer(LANGUAGE)
    summarizer = TextRankSummarizer(stemmer) 
    summarizer.stop_words = get_stop_words(LANGUAGE)
    # print("TextRank summarizer loaded successfully.") # Console output handled below
except Exception as e:
    summarizer = None

# --- Global Deep Learning Model Setup (Whisper STT) ---
try:
    print("Loading Whisper STT model (this may take a moment)...")
    stt_model = whisper.load_model("base") 
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to load Whisper STT model: {e}")
    stt_model = None
    
if summarizer is None:
    print("CRITICAL: TextRank summarizer failed to load.")
else:
    print("TextRank summarizer loaded successfully.")

# --- HELPER FUNCTIONS (FIXED PLACEMENT) ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_summary(summary_list):
    """Applies punctuation and bullet points universally."""
    formatted_points = []
    for sentence in summary_list:
        sentence = str(sentence).strip()
        if sentence and sentence[-1] not in ['.', '!', '?']:
            sentence += '.'
        formatted_points.append(f'* {sentence}')
    return "\n".join(formatted_points)

def summarize_text_content(text_content, ratio=SENTENCES_COUNT_RATIO):
    """
    General function to summarize raw text content using TextRank.
    """
    if summarizer is None:
         raise RuntimeError("Text Summarizer is not available.")
    
    all_sentences = sent_tokenize(text_content)
    total_sentences = len(all_sentences)
    target_sentence_count = max(5, int(total_sentences * ratio))

    parser = PlaintextParser.from_string(text_content, Tokenizer(LANGUAGE))
    summary_sentences = summarizer(parser.document, target_sentence_count)
    
    clean_summary_list = []
    for sentence in summary_sentences:
        s = str(sentence)
        normalized_sentence = s.lower().strip()
        if len(normalized_sentence.split()) < 7:
             continue
        if any(word in normalized_sentence for word in ['email:', 'http', 'https', 'figure', 'table', 'source:']):
             continue 
        
        clean_summary_list.append(s)

    return format_summary(clean_summary_list)

def extract_text_from_pdf_or_txt(media_path):
    """Extracts text from a PDF or TXT file."""
    if media_path.lower().endswith('.txt'):
        with open(media_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    text = ""
    try:
        doc = fitz.open(media_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise Exception(f"Could not extract text from file: {e}")


def extract_transcript(media_path):
    """
    Extracts text transcript from an audio or video file using Whisper.
    """
    if stt_model is None:
        raise RuntimeError("Speech-to-Text model is not available.")
    
    print(f"Starting Whisper transcription for {os.path.basename(media_path)}...")
    result = stt_model.transcribe(media_path, fp16=False) 
    
    transcript = result["text"]
    if not transcript:
        raise ValueError("Could not extract any meaningful speech from the media file. Check audio quality.")
        
    return transcript

def summarize_multimedia(media_path):
    """Handles both audio and video files."""
    transcript = extract_transcript(media_path)
    summary = summarize_text_content(transcript) 
    
    return summary

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_file():
    # 1. Handle File Upload and Validation
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # *** NameError FIX: allowed_file is defined globally above ***
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: PDF, TXT, MP4, AVI, MP3, WAV, MOV."}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath) 
    
    file_extension = filename.rsplit('.', 1)[1].lower()
    summary = ""
    status = "success"

    try:
        # 2. Run Summarization based on file type
        if file_extension in ['pdf', 'txt']:
            print(f"Starting document summarization for {file_extension}...")
            text_content = extract_text_from_pdf_or_txt(filepath)
            summary = summarize_text_content(text_content)
        elif file_extension in ['mp4', 'avi', 'mov', 'mp3', 'wav']:
            # This is the heavy step handled by the multimodal function
            summary = summarize_multimedia(filepath)
            
    except RuntimeError as e:
        status = "error"
        summary = f"Model Error: {str(e)}"
    except Exception as e:
        status = "error"
        print("\n--- CRITICAL ERROR TRACEBACK ---")
        traceback.print_exc()
        summary = f"An internal processing error occurred: {type(e).__name__} - {str(e)}"
        
    finally:
        if os.path.exists(filepath):
             os.remove(filepath)

    return jsonify({"status": status, "summary": summary, "original_filename": filename})

if __name__ == '__main__':
    # WARNING: Set threaded=False for stability during heavy Whisper tasks
    app.run(debug=True, threaded=False)
# pip install nltk
# pip install sumy nltk