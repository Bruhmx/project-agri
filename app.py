import os
from datetime import datetime
import json
import sys
import re
import time

from flask import Flask, render_template, request, session, jsonify, flash, url_for, make_response, Response, send_file
from werkzeug.utils import secure_filename, redirect, send_from_directory

from auth import login_required
from db_config import get_db, get_db_cursor, get_db_cursor_readonly, return_db, close_all_connections
from predictor import predict_crop, predict_disease, get_crop_display_name, get_disease_display_name
from user_routes import register_user_routes

import user_routes
import atexit

app = Flask(__name__, static_folder='static')

# Production configuration
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Use environment variables for paths
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload directories exist with proper paths
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static/uploads/feedback'), exist_ok=True)
os.makedirs(os.path.join('static/uploads/profiles'), exist_ok=True)

# Register cleanup on shutdown
atexit.register(close_all_connections)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ========== DATABASE SAVE FUNCTIONS ==========
def save_initial_diagnosis(user_id, image_file, crop, disease_data):
    """Save initial AI diagnosis to database with image path"""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(image_file.filename)

        if '.' in original_filename:
            file_extension = original_filename.rsplit('.', 1)[1].lower()
        else:
            file_extension = 'jpg'

        # Create new filename
        new_filename = f"user_{user_id}_{timestamp}_{original_filename}"

        # Full path to save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

        # Save the file to disk
        image_file.seek(0)
        image_file.save(file_path)
        print(f"✅ Image saved to disk: {file_path}")

        # Store just the filename in database
        db_image_path = new_filename

        # Combine treatment info into recommendations
        recommendations = f"Manual: {disease_data.get('manual_treatment', 'N/A')}. "
        recommendations += f"Organic: {disease_data.get('organic_treatment', 'N/A')}. "
        recommendations += f"Chemical: {disease_data.get('chemical_treatment', 'N/A')}."

        # Use context manager for database operation
        with get_db_cursor() as cursor:
            query = """
            INSERT INTO diagnosis_history 
            (user_id, image_path, crop, disease_detected, 
             confidence, symptoms, recommendations, for_training)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            values = (
                user_id,
                db_image_path,
                crop,
                disease_data['name'],
                disease_data['confidence'],
                disease_data.get('symptoms', ''),
                recommendations,
                True
            )

            cursor.execute(query, values)
            diagnosis_id = cursor.fetchone()[0]

        file_size = os.path.getsize(file_path)
        print(f"✅ Diagnosis saved with ID: {diagnosis_id}")
        print(f"✅ Image path stored in DB: '{db_image_path}'")
        print(f"✅ File saved at: {file_path} (Size: {file_size} bytes)")
        return diagnosis_id

    except Exception as e:
        print(f"❌ Error saving initial diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return None

def update_diagnosis_with_answers(diagnosis_id, answers_data, summary_data):
    """Update diagnosis with expert answers and summary"""
    try:
        with get_db_cursor() as cursor:
            cursor.execute("""
                UPDATE diagnosis_history 
                SET expert_answers = %s::jsonb,
                    expert_summary = %s::jsonb,
                    final_confidence_level = %s
                WHERE id = %s
            """, (
                json.dumps(answers_data),
                json.dumps(summary_data),
                summary_data.get('confidence', 'Possible'),
                diagnosis_id
            ))
        return True
    except Exception as e:
        print(f"❌ Error updating diagnosis: {e}")
        return False


def save_exported_training_data(diagnosis_ids):
    """Mark images as used in training"""
    if not diagnosis_ids:
        return

    try:
        with get_db_cursor() as cursor:
            placeholders = ','.join(['%s'] * len(diagnosis_ids))
            cursor.execute(f"""
                UPDATE diagnosis_history 
                SET training_used = TRUE,
                    image_processed = TRUE
                WHERE id IN ({placeholders})
            """, diagnosis_ids)
        return True
    except Exception as e:
        print(f"❌ Error marking training data: {e}")
        return False


# ========== IMAGE SERVING ROUTE ==========
@app.route('/diagnosis-image/<int:diagnosis_id>')
@login_required
def diagnosis_image(diagnosis_id):
    """Serve diagnosis image from file system"""
    user_id = session['user_id']
    is_admin = session.get('is_admin', False)

    try:
        # Use context manager for readonly operation
        with get_db_cursor_readonly() as cursor:
            if is_admin:
                cursor.execute("SELECT image_path FROM diagnosis_history WHERE id = %s", (diagnosis_id,))
            else:
                cursor.execute("SELECT image_path FROM diagnosis_history WHERE id = %s AND user_id = %s",
                            (diagnosis_id, user_id))

            result = cursor.fetchone()

        if result and result['image_path']:
            image_path = result['image_path']
            print(f"Image path from DB: '{image_path}'")

            # Get app directory
            app_dir = os.path.dirname(os.path.abspath(__file__))

            # Get just the filename (remove any path prefixes)
            filename = os.path.basename(image_path)

            # Always look in static/uploads/ folder
            full_path = os.path.join(app_dir, 'static', 'uploads', filename)

            print(f"Looking for image at: {full_path}")

            if os.path.exists(full_path):
                print(f"✅ Found image at: {full_path}")

                # Determine mimetype
                ext = os.path.splitext(full_path)[1].lower()
                mimetypes = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                mimetype = mimetypes.get(ext, 'image/jpeg')

                return send_file(full_path, mimetype=mimetype)
            else:
                print(f"❌ File does not exist at: {full_path}")

        print(f"❌ No image found for diagnosis {diagnosis_id}")
        return send_placeholder_image()

    except Exception as e:
        print(f"Error in diagnosis_image: {e}")
        import traceback
        traceback.print_exc()
        return send_placeholder_image()

def send_placeholder_image():
    """Helper function to send a placeholder image"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    placeholder_paths = [
        os.path.join(app_dir, 'static', 'images', 'placeholder.jpg'),
        os.path.join(app_dir, 'static', 'img', 'placeholder.jpg'),
        os.path.join(app_dir, 'static', 'placeholder.jpg')
    ]

    for path in placeholder_paths:
        if os.path.exists(path):
            return send_file(path, mimetype='image/jpeg')

    # If no placeholder found, create a simple colored image on the fly
    try:
        from PIL import Image, ImageDraw
        import io

        # Create a simple image
        img = Image.new('RGB', (400, 300), color='#f0f0f0')
        d = ImageDraw.Draw(img)
        d.text((150, 150), "No Image Available", fill='#999999')

        # Save to bytes
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')
    except ImportError:
        # If PIL is not available, return a 404
        return "Image not found", 404


# ========== DEBUG AND TEST ROUTES ==========

@app.route("/debug-env")
def debug_environment():
    """Debug endpoint to check environment variables"""
    # Get database URL (mask password for security)
    db_url = os.environ.get('DATABASE_URL', 'Not set')
    if db_url != 'Not set':
        # Mask the password part for display
        masked_url = re.sub(r':([^@]+)@', ':****@', db_url)
    else:
        masked_url = 'Not set'
    
    return jsonify({
        "success": True,
        "database_url_set": os.environ.get('DATABASE_URL') is not None,
        "database_url": masked_url,
        "secret_key_set": os.environ.get('SECRET_KEY') is not None,
        "flask_env": os.environ.get('FLASK_ENV', 'not set'),
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "upload_folder_exists": os.path.exists(app.config['UPLOAD_FOLDER']),
        "environment_vars": [k for k in os.environ.keys() if not k.startswith('DATABASE')][:5]
    })


@app.route("/test-db-connection")
def test_db_connection():
    """Test if we can connect to the PostgreSQL database"""
    try:
        # Use context manager for test
        with get_db_cursor_readonly() as cursor:
            cursor.execute("SELECT current_database(), current_user, version();")
            result = cursor.fetchone()
        
        return jsonify({
            "success": True,
            "database": result[0],
            "user": result[1],
            "version": result[2][:50] + "...",
            "message": "✅ Successfully connected to PostgreSQL!"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }), 500


@app.route("/init-db")
def init_database_route():
    """One-time route to initialize database tables"""
    secret = request.args.get('secret')
    expected_secret = os.environ.get('INIT_SECRET', 'agriaid-init-2024')
    
    if secret != expected_secret:
        return jsonify({
            "success": False,
            "error": "Unauthorized - Invalid secret"
        }), 401
    
    try:
        from init_postgres_db import init_postgres_database
        result = init_postgres_database()
        
        return jsonify({
            "success": True,
            "message": "✅ Database initialized successfully!",
            "details": result
        })
    except ImportError:
        return jsonify({
            "success": False,
            "error": "init_postgres_db.py not found. Please create it first."
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/check-tables")
def check_tables():
    """List all tables in the database"""
    try:
        with get_db_cursor_readonly() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = cursor.fetchall()
            
            # Get row counts for each table
            table_stats = []
            for table in tables:
                table_name = table['table_name']
                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                count = cursor.fetchone()['count']
                table_stats.append({
                    'name': table_name,
                    'rows': count
                })
        
        return jsonify({
            "success": True,
            "tables": table_stats,
            "total_tables": len(table_stats)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/health")
def health_check():
    """Health check endpoint for Render"""
    try:
        # Quick database check
        with get_db_cursor_readonly() as cursor:
            cursor.execute("SELECT 1 as health_check")
            result = cursor.fetchone()
        
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }), 500


# ========== MAIN ROUTES ==========

@app.route("/")
def index():
    """Home page with system description"""
    # Only clear diagnosis data if needed
    diagnosis_keys = [
        'crop', 'crop_display', 'crop_confidence',
        'diseases', 'question_tree', 'all_questions_flat',
        'user_answers', 'pending_diagnosis', 'pending_questions'
    ]

    for key in diagnosis_keys:
        session.pop(key, None)

    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_image():
    """Handle image upload and show AI diagnosis results"""
    if request.method == "POST":
        # Check if file was uploaded
        if 'image' not in request.files:
            return render_template("upload.html", error="No file selected")

        file = request.files['image']

        if file.filename == '':
            return render_template("upload.html", error="No file selected")

        if not allowed_file(file.filename):
            return render_template("upload.html", error="File type not allowed. Please upload an image.")

        # Store file position for later use
        file.seek(0)

        # Create a temporary file for AI processing
        temp_filename = secure_filename(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_filepath)

        try:
            # Step 1: Crop Identification
            crop, crop_conf = predict_crop(temp_filepath)

            # Validate crop detection
            valid_crops = ['corn', 'rice']
            if crop not in valid_crops or float(crop_conf) < 0.8:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                return render_template("upload.html",
                                       error="Invalid image - Please upload a clear image of corn or rice leaves.")

            # Step 2: Disease Prediction
            diseases = predict_disease(temp_filepath, crop)

            # Store in session
            session['crop'] = crop
            session['crop_display'] = get_crop_display_name(crop)

            # Get disease details from database for all detected diseases
            disease_results = []
            
            # Use context manager for database operations
            with get_db_cursor() as cur:
                # Prepare results for all top diseases
                for disease_name, confidence in diseases[:3]:
                    # Get disease information
                    cur.execute("""
                        SELECT * FROM disease_info 
                        WHERE crop = %s AND disease_code = %s
                    """, (crop, disease_name))

                    disease_details = cur.fetchone()

                    # Get sample images
                    cur.execute("""
                        SELECT id, image_title as title, 
                               image_description as description, severity_level as severity
                        FROM disease_samples 
                        WHERE crop = %s AND disease_code = %s 
                        ORDER BY display_order
                    """, (crop, disease_name))

                    sample_images = cur.fetchall()
                    # Generate URLs for each sample
                    for sample in sample_images:
                        sample['url'] = url_for('get_disease_sample_image', sample_id=sample['id'])

                    disease_result = {
                        'code': disease_name,
                        'name': get_disease_display_name(disease_name),
                        'confidence': float(confidence) * 100,
                        'cause': disease_details.get('cause', 'Information not available') if disease_details else 'Information not available',
                        'symptoms': disease_details.get('symptoms', 'Symptoms information not available') if disease_details else 'Symptoms information not available',
                        'manual_treatment': disease_details.get('manual_treatment', 'Remove affected leaves and maintain proper spacing.') if disease_details else 'Remove affected leaves and maintain proper spacing.',
                        'organic_treatment': disease_details.get('organic_treatment', 'Apply neem oil or baking soda solution.') if disease_details else 'Apply neem oil or baking soda solution.',
                        'chemical_treatment': disease_details.get('chemical_treatment', 'Consult with agricultural expert for chemical recommendations.') if disease_details else 'Consult with agricultural expert for chemical recommendations.',
                        'prevention': disease_details.get('prevention', 'Practice crop rotation and maintain field hygiene.') if disease_details else 'Practice crop rotation and maintain field hygiene.',
                        'sample_images': sample_images
                    }
                    disease_results.append(disease_result)

            # ===== SAVE TO DATABASE WITH IMAGE =====
            user_id = session.get('user_id')
            if user_id and disease_results:
                # Go back to beginning of file for reading
                file.seek(0)

                diagnosis_id = save_initial_diagnosis(
                    user_id=user_id,
                    image_file=file,
                    crop=crop,
                    disease_data=disease_results[0]
                )

                if diagnosis_id:
                    session['current_diagnosis_id'] = diagnosis_id
                    print(f"✅ Diagnosis saved with ID: {diagnosis_id}")
                else:
                    print("⚠️ Failed to save diagnosis to database")

            # Clean up temp file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

            # Store AI results in session
            session['ai_diagnosis'] = {
                'primary': disease_results[0],
                'alternatives': disease_results[1:] if len(disease_results) > 1 else [],
                'crop_original': crop,
                'crop': session['crop_display']
            }

            # Render results page with AI diagnosis
            return render_template("ai_results.html",
                                   diagnosis=session['ai_diagnosis'],
                                   diagnosis_id=session.get('current_diagnosis_id', 0))

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            # Clean up temp file if it exists
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return render_template("upload.html", error="Error processing image. Please try again.")

    return render_template("upload.html")


@app.route("/optional-questions/<disease_code>")
@login_required
def optional_questions(disease_code):
    """Optional expert questions for additional information"""
    if 'ai_diagnosis' not in session:
        return redirect("/upload")

    crop = session.get('crop')

    # Get questions for this disease
    with get_db_cursor() as cur:
        cur.execute("""
            SELECT q.id, q.question_text, q.yes_score, q.no_score, 
                   q.question_category, q.priority, q.depends_on, q.show_if_answer,
                   COALESCE(di.disease_name, q.target) as disease_name
            FROM questions q
            LEFT JOIN disease_info di ON q.crop = di.crop AND q.target = di.disease_code
            WHERE q.crop = %s AND q.target = %s
            ORDER BY q.priority, q.id
        """, (crop, disease_code))

        all_questions = cur.fetchall()

    # Get root questions (no dependencies)
    root_questions = [q for q in all_questions if q['depends_on'] is None]

    return render_template("optional_questions.html",
                           disease_code=disease_code,
                           disease_name=get_disease_display_name(disease_code),
                           crop=session.get('crop_display'),
                           questions=root_questions,
                           all_questions=all_questions,
                           diagnosis_id=session.get('current_diagnosis_id', 0))


# [Keep all the other routes the same - they already use context managers correctly]

# Register user routes
register_user_routes(app)

# ========== ADDITIONAL ROUTES ==========

@app.route("/about")
def about():
    """System description page"""
    return render_template("about.html")


@app.route("/api/disease-info")
def get_disease_info():
    """API endpoint to get detailed disease information"""
    crop = request.args.get('crop')
    disease_code = request.args.get('disease')

    if not crop or not disease_code:
        return jsonify({'success': False, 'error': 'Missing parameters'})

    try:
        with get_db_cursor() as cur:
            # Get disease information
            cur.execute("""
                SELECT * FROM disease_info 
                WHERE crop = %s AND disease_code = %s
            """, (crop, disease_code))

            disease_data = cur.fetchone()

            if not disease_data:
                return jsonify({'success': False, 'error': 'Disease not found'})

            # Get all sample images for this disease
            cur.execute("""
                SELECT id, image_title as title, 
                       image_description as description, severity_level as severity
                FROM disease_samples 
                WHERE crop = %s AND disease_code = %s 
                ORDER BY display_order
            """, (crop, disease_code))

            sample_images = cur.fetchall()

            # Generate URLs for each sample
            for sample in sample_images:
                sample['url'] = url_for('get_disease_sample_image', sample_id=sample['id'])

        return jsonify({
            'success': True,
            'disease_name': get_disease_display_name(disease_code),
            'crop_display': get_crop_display_name(crop),
            'cause': disease_data.get('cause', 'Information not available'),
            'symptoms': disease_data.get('symptoms', 'No symptoms described'),
            'organic_treatment': disease_data.get('organic_treatment', 'Not specified'),
            'chemical_treatment': disease_data.get('chemical_treatment', 'Not specified'),
            'prevention': disease_data.get('prevention', 'Not specified'),
            'manual_treatment': disease_data.get('manual_treatment', 'Not specified'),
            'image_url': sample_images[0]['url'] if sample_images else url_for('static', filename='img/disease-placeholder.jpg'),
            'sample_images': sample_images,
            'last_updated': disease_data.get('created_at')
        })

    except Exception as e:
        print(f"Error getting disease info: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/disease-library')
@login_required
def disease_library():
    """Public disease library page"""
    try:
        crop = request.args.get('crop', 'corn')

        with get_db_cursor() as cur:
            cur.execute("""
                SELECT 
                    di.id,
                    di.disease_code,
                    di.crop,
                    di.cause,
                    di.symptoms,
                    di.organic_treatment,
                    di.chemical_treatment,
                    di.prevention,
                    di.manual_treatment,
                    di.created_at,
                    (SELECT COUNT(*) FROM disease_samples 
                     WHERE disease_code = di.disease_code AND crop = di.crop) as sample_count,
                    (SELECT id FROM disease_samples 
                     WHERE disease_code = di.disease_code AND crop = di.crop 
                     ORDER BY display_order LIMIT 1) as first_sample_id
                FROM disease_info di
                WHERE di.crop = %s
                ORDER BY di.disease_code
            """, (crop,))

            diseases = cur.fetchall()

        # Create image URLs for each disease using the first sample
        for disease in diseases:
            if disease['first_sample_id']:
                disease['sample_image'] = url_for('get_disease_sample_image', sample_id=disease['first_sample_id'])
            else:
                disease['sample_image'] = url_for('static', filename='img/disease-placeholder.jpg')

        crop_display = 'Corn' if crop == 'corn' else 'Rice'

        return render_template('disease_library.html',
                               diseases=diseases,
                               crop=crop,
                               crop_display=crop_display)

    except Exception as e:
        print(f"Error in disease_library: {e}")
        import traceback
        traceback.print_exc()
        flash('Error loading disease library', 'danger')
        return render_template('disease_library.html', diseases=[], crop='corn', crop_display='Corn')


# [Keep all the remaining routes exactly as they are - they already use context managers]

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template("500.html"), 500


def create_placeholders():
    """Create placeholder images if they don't exist"""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(app_dir, 'static', 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Create a simple 1x1 transparent pixel as placeholder
    import base64
    pixel = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')

    no_image_path = os.path.join(img_dir, 'no-image.png')
    if not os.path.exists(no_image_path):
        with open(no_image_path, 'wb') as f:
            f.write(pixel)
        print(f"✅ Created placeholder: {no_image_path}")

    error_image_path = os.path.join(img_dir, 'error-image.png')
    if not os.path.exists(error_image_path):
        with open(error_image_path, 'wb') as f:
            f.write(pixel)
        print(f"✅ Created placeholder: {error_image_path}")


if __name__ == "__main__":
    create_placeholders()
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    # Run with host='0.0.0.0' to accept external connections
    app.run(host='0.0.0.0', port=port)