import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np
import os
import gc
import re
import time
import sys
import json
import argparse
import uuid
from merge_lines import merge_text_lines


from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- MODEL INITIALIZATION ---
gemini_model = None
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if gemini_api_key:
        from google import generativeai as genai
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        print("[INFO] Google Gemini AI model initialized successfully (gemini-1.5-flash-latest)", file=sys.stderr)
    else:
        print("[WARNING] GOOGLE_API_KEY not found. AI features will be disabled.", file=sys.stderr)
except Exception as e:
    print(f"[ERROR] Failed to initialize Google Gemini AI: {str(e)}", file=sys.stderr)
    gemini_model = None

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[INFO] PyTorch using device: {torch_device}", file=sys.stderr)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

ocr = None
try:
    import paddle
    from paddleocr import PaddleOCR
    use_gpu_paddle = paddle.device.is_compiled_with_cuda()
    if use_gpu_paddle:
        paddle.device.set_device('gpu')
        print("[PaddleOCR will use GPU", file=sys.stderr)
    else:
        paddle.device.set_device('cpu')
        print("PaddleOCR will use CPU", file=sys.stderr)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, det_db_box_thresh=0.4)

    print("PaddleOCR initialized successfully", file=sys.stderr)
except Exception as e:
    print(f"Error initializing PaddleOCR: {str(e)}", file=sys.stderr)
    ocr = None

processor_trocr = None
model_trocr = None
try:
    processor_trocr = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten', cache_dir='model_cache')
    model_trocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten', cache_dir='model_cache')
    model_trocr = model_trocr.to(torch_device)
    model_trocr.eval()
    if torch_device == 'cuda': print("[INFO] TrOCR using GPU", file=sys.stderr)
    print("TrOCR initialized successfully", file=sys.stderr)
except Exception as e:
    print(f"Error initializing TrOCR: {str(e)}", file=sys.stderr)
    processor_trocr = model_trocr = None

def print_progress(message: str):
    print(f"[PROGRESS-API]: {message}", file=sys.stderr)
    sys.stderr.flush()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'^\.+', '', text)
    return text

def correct_text_with_ai(text_to_correct: str) -> str:
    if not gemini_model or not text_to_correct:
        print("AI Correction: Gemini model not available or no text provided.", file=sys.stderr)
        return text_to_correct
    try:
        print_progress("AI: Correcting text...")
        prompt = f"""Correct the spelling and grammar of the following text. Preserve the original meaning and structure as much as possible. Only output the corrected text, with no additional explanations, preambles, or conversational filler:

Original Text:
"{text_to_correct}"

Corrected Text:
"""
        response = gemini_model.generate_content(prompt)
        corrected_text = response.text.strip() if hasattr(response, 'text') else ""
        if corrected_text:
            print_progress("AI: Correction successful.")
            return corrected_text
        else:
            print("[WARNING] AI model did not return corrected text, returning original.", file=sys.stderr)
            return text_to_correct
    except Exception as e:
        print(f"[ERROR] Failed to correct text with AI: {str(e)}", file=sys.stderr)
        return text_to_correct

def summarise_text_with_ai(text_to_summarise: str) -> str:
    if not gemini_model or not text_to_summarise:
        print("AI Summarisation: Gemini model not available or no text provided.", file=sys.stderr)
        return "AI summarisation not available or no text to summarise."
    try:
        print_progress("AI: Summarising text...")
        prompt = f"""Provide a concise summary of the following text:

"{text_to_summarise}"

Summary:
"""
        response = gemini_model.generate_content(prompt)
        summary = response.text.strip() if hasattr(response, 'text') else ""
        if summary:
            print_progress("AI: Summarization successful.")
            return summary
        else:
            print("[WARNING] AI model did not return a summary.", file=sys.stderr)
            return "Failed to generate summary."
    except Exception as e:
        print(f"[ERROR] Failed to summarise text with AI: {str(e)}", file=sys.stderr)
        return "Failed to generate summary."

def merge_paddle_boxes_to_lines(
    paddle_detections: list, y_overlap_threshold_ratio: float = 0.4, 
    max_height_ratio_filter: float = 3.0, min_box_height_px: int = 5
) -> list[list[list[int]]]:
    """
    Use the improved merge_text_lines function from merge_lines.py
    """
    if not paddle_detections: 
        return []
    
    # Convert paddle_detections to the format expected by merge_text_lines
    # merge_text_lines expects results[0] format from PaddleOCR
    results = [paddle_detections]  # Wrap in list to match expected format
    
    # Use the imported merge_text_lines function with default parameters
    # You can adjust x_tolerance and y_tolerance_scale as needed
    merged_boxes = merge_text_lines(results, x_tolerance=30, y_tolerance_scale=0.5)
    
    # Convert numpy float values to integers for C# compatibility
    converted_boxes = []
    for box in merged_boxes:
        converted_box = []
        for point in box:
            converted_point = [int(coord) for coord in point]
            converted_box.append(converted_point)
        converted_boxes.append(converted_box)
    
    return converted_boxes

def preprocess_image_for_ocr(cv2_image_bgr):
    print_progress("Applying image preprocessing (if any steps are uncommented)...")
    processed_image = cv2_image_bgr.copy()
    return processed_image

def extract_text_and_boxes(image_path: str, use_improvement: bool = False) -> dict:
    if ocr is None: return {"success": False, "error": "PaddleOCR not initialized."}
    if processor_trocr is None or model_trocr is None: return {"success": False, "error": "TrOCR not initialized."}
    start_time = time.time()
    try:
        print_progress(f"Processing image: {image_path}")
        image_cv_bgr = cv2.imread(image_path)
        if image_cv_bgr is None: return {"success": False, "error": "Failed to read image"}
        
        preprocessed_image_cv_bgr = preprocess_image_for_ocr(image_cv_bgr)
        input_for_paddle = preprocessed_image_cv_bgr

        paddle_result = ocr.ocr(input_for_paddle, cls=True)
        if not paddle_result or not paddle_result[0]: return {"success": False, "error": "No text detected by PaddleOCR"}
        
        raw_paddle_detections = paddle_result[0]
        merged_line_boxes = merge_paddle_boxes_to_lines(raw_paddle_detections)
        if not merged_line_boxes: return {"success": False, "error": "No lines formed after merging boxes"}

        processed_texts_per_line = []
        if len(preprocessed_image_cv_bgr.shape) == 2 or preprocessed_image_cv_bgr.shape[2] == 1:
            image_for_trocr_pil = Image.fromarray(cv2.cvtColor(preprocessed_image_cv_bgr, cv2.COLOR_GRAY2RGB))
        else:
            image_for_trocr_pil = Image.fromarray(cv2.cvtColor(preprocessed_image_cv_bgr, cv2.COLOR_BGR2RGB))

        for i, box_points in enumerate(merged_line_boxes, 1):
            x_coords = [p[0] for p in box_points]; y_coords = [p[1] for p in box_points]
            x_min,y_min = int(min(x_coords)),int(min(y_coords)); x_max,y_max = int(max(x_coords)),int(max(y_coords))
            w_pil, h_pil = image_for_trocr_pil.size
            x_min_c,y_min_c=max(0,x_min),max(0,y_min); x_max_c,y_max_c=min(w_pil,x_max),min(h_pil,y_max)
            if x_max_c <= x_min_c or y_max_c <= y_min_c: processed_texts_per_line.append("(empty region)"); continue
            pil_crop = image_for_trocr_pil.crop((x_min_c, y_min_c, x_max_c, y_max_c))
            if pil_crop.size[0] == 0 or pil_crop.size[1] == 0: processed_texts_per_line.append("(empty crop)"); continue
            
            with torch.no_grad():
                pixel_values = processor_trocr(pil_crop, return_tensors="pt").pixel_values.to(torch_device)
                if torch_device == 'cuda' and hasattr(model_trocr.config, 'torch_dtype') and model_trocr.config.torch_dtype == torch.float16:
                     pixel_values = pixel_values.half()
                generated_ids = model_trocr.generate(pixel_values, max_length=128)
                transcription = processor_trocr.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_texts_per_line.append(clean_text(transcription))
        
        final_text = "\n".join(filter(None, processed_texts_per_line))
        corrected_text, summary_text = final_text, ""
        if use_improvement and final_text and gemini_model:
            print_progress("Attempting AI text correction...")
            corrected_text_from_ai = correct_text_with_ai(final_text)
            if corrected_text_from_ai != final_text: corrected_text = corrected_text_from_ai
            if corrected_text:
                print_progress("Attempting AI text summarisation...")
                summary_from_ai = summarise_text_with_ai(corrected_text)
                if summary_from_ai and "Failed" not in summary_from_ai and "not available" not in summary_from_ai:
                    summary_text = summary_from_ai
        elif use_improvement and not gemini_model:
             print("[WARNING] AI improvement requested, but Gemini model is not available.", file=sys.stderr)

        return {
            "success": True, "texts_per_line": processed_texts_per_line,
            "original_text": final_text, "corrected_text": corrected_text,
            "summary_text": summary_text, "boxes": merged_line_boxes
        }
    except Exception as e:
        import traceback; tb_str = traceback.format_exc()
        print(f"[ERROR] API: Critical error: {str(e)}\n{tb_str}", file=sys.stderr)
        return {"success": False, "error": f"Server error: {str(e)}"}
    finally:
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

# --- FLASK API ENDPOINTS ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/recognize', methods=['POST'])
def recognize_image_endpoint():
    if ocr is None or processor_trocr is None or model_trocr is None:
        return jsonify({"success": False, "error": "OCR models not initialized. Check server logs."}), 503
    if 'image_file' not in request.files:
        return jsonify({'success': False, 'error': 'No image_file part in request'}), 400
    file = request.files['image_file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename): # Call to allowed_file
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try:
            file.save(temp_image_path)
            use_improvement_str = request.form.get('use_improvement', 'false')
            use_improvement_bool = use_improvement_str.lower() == 'true'
            result = extract_text_and_boxes(temp_image_path, use_improvement_bool)
            return jsonify(result)
        except Exception as e:
            import traceback; tb_str = traceback.format_exc()
            print(f"[ERROR] API: Error during file processing or OCR: {str(e)}\n{tb_str}", file=sys.stderr)
            return jsonify({'success': False, 'error': f'Server processing error: {str(e)}'}), 500
        finally:
            if os.path.exists(temp_image_path):
                try: os.remove(temp_image_path)
                except Exception as e_del: print(f"[ERROR] API: Failed to delete temp file '{temp_image_path}': {str(e_del)}", file=sys.stderr)
    else:
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    errors = []
    if ocr is None: errors.append("PaddleOCR not initialized.")
    if processor_trocr is None or model_trocr is None: errors.append("TrOCR not initialized.")
    if gemini_model is None and os.getenv("GOOGLE_API_KEY"): errors.append("Gemini AI model not initialized (API key was present).")
    status = "OK" if not errors else ("DEGRADED" if (ocr or model_trocr) else "ERROR")
    http_status_code = 200 if status == "OK" else 503
    return jsonify({
        "status": status, 
        "models_initialized": {
            "paddleocr": ocr is not None, 
            "trocr": model_trocr is not None and processor_trocr is not None, 
            "gemini": gemini_model is not None
        }, 
        "details": errors
    }), http_status_code

if __name__ == "__main__":
    print("[INFO] Starting Flask server for HTRex OCR service...", file=sys.stderr)
    app.run(host='0.0.0.0', port=5000, debug=True)