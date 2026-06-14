import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import your validated backend logic
from src.retriever import retrieve_similar_cases
from src.gemini_bridge import analyze_skin_condition

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    """Renders the main dashboard interface."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint handling image upload, vector search, and Gemini synthesis."""
    # 1. Validate incoming file
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400
        
    file = request.files['file']
    region = request.form.get('region', 'cheeks') # Defaults to cheeks if not provided
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected for uploading'}), 400
        
    if file and allowed_file(file.filename):
        # Secure the filename and save to static uploads
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 2. Query ChromaDB via your Retrieval Engine
            print(f"[API] Querying vector index for region: {region}")
            matched_cases = retrieve_similar_cases(filepath, region_name=region, top_k=3)
            
            # 3. Request synthesized report from Gemini 2.5
            print("[API] Submitting multimodal context payload to Gemini...")
            ai_analysis = analyze_skin_condition(filepath, matched_cases)
            
            # 4. Format metadata safely for json response
            formatted_matches = []
            if matched_cases and 'metadatas' in matched_cases and len(matched_cases['metadatas'][0]) > 0:
                for i in range(len(matched_cases['metadatas'][0])):
                    formatted_matches.append({
                        'class': matched_cases['metadatas'][0][i]['class_label'],
                        'similarity': round((1.0 - matched_cases['distances'][0][i]) * 100, 1)
                    })

            return jsonify({
                'success': True,
                'image_url': f'/{filepath}',
                'analysis': ai_analysis,
                'matches': formatted_matches
            })
            
        except Exception as e:
            print(f"[API ERROR] Pipeline execution failed: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
            
    return jsonify({'success': False, 'error': 'Allowed file types are png, jpg, jpeg, webp'}), 400

if __name__ == '__main__':
    # Running on port 7860 to match Hugging Face Space default expectations
    app.run(host='0.0.0.0', port=7860, debug=True)