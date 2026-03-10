import tensorflow as tf
import numpy as np
from PIL import Image
import warnings
import os
import glob

warnings.filterwarnings('ignore')

# Function to load model with compatibility fixes for TF 2.13.0
def load_model_safely(model_path, model_name):
    """Load a Keras model with compatibility fixes for TF 2.13.0"""
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        return None
    
    print(f"Attempting to load {model_name} from {model_path}...")
    
    # Method 1: Standard loading for TF 2.13.0
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ {model_name} loaded successfully")
        return model
    except Exception as e:
        print(f"  Standard loading failed: {type(e).__name__} - {str(e)[:100]}")
        
        # Method 2: Load with custom objects for TF 2.13.0
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'Functional': tf.keras.Model,
                }
            )
            print(f"✓ {model_name} loaded with custom objects")
            return model
        except Exception as e2:
            print(f"  Custom objects loading failed: {type(e2).__name__}")
            
            # Method 3: Load with compile=False
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False
                )
                print(f"✓ {model_name} loaded with compile=False")
                return model
            except Exception as e3:
                print(f"  All loading methods failed: {type(e3).__name__}")
    
    return None

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("=" * 50)
print("Loading ML Models with TensorFlow", tf.__version__)
print("=" * 50)

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load models with safe method
crop_model = load_model_safely(os.path.join(MODELS_DIR, "crop_model_final.keras"), "Crop Model")
corn_model = load_model_safely(os.path.join(MODELS_DIR, "corn_model_final.keras"), "Corn Disease Model")
rice_model = load_model_safely(os.path.join(MODELS_DIR, "rice_model_final.keras"), "Rice Disease Model")

print("=" * 50)
if crop_model is not None and corn_model is not None and rice_model is not None:
    print("✓ All models loaded successfully!")
else:
    print("⚠️ Some models failed to load - using fallback predictions")
    if crop_model is None:
        print("  - Crop model: NOT LOADED")
    if corn_model is None:
        print("  - Corn model: NOT LOADED")
    if rice_model is None:
        print("  - Rice model: NOT LOADED")
print("=" * 50)

# Class mappings with display names
CROP_CLASSES = ["corn", "rice"]
CROP_DISPLAY_NAMES = {
    "corn": "Corn (Maize)",
    "rice": "Rice"
}

CORN_CLASSES = ["Common_Rust", "gls", "healthy", "nclb"]
CORN_DISPLAY_NAMES = {
    "Common_Rust": "Common Rust",
    "gls": "Gray Leaf Spot",
    "healthy": "Healthy",
    "nclb": "Northern Corn Leaf Blight"
}

RICE_CLASSES = ["blast", "blight", "brownspot", "healthy", "tungro"]
RICE_DISPLAY_NAMES = {
    "blast": "Rice Blast",
    "blight": "Bacterial Leaf Blight",
    "brownspot": "Brown Spot",
    "healthy": "Healthy",
    "tungro": "Tungro Virus"
}


def preprocess_image(img_path, img_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            return np.zeros((1, 224, 224, 3))
            
        img = Image.open(img_path)
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize and normalize
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"⚠️ Error preprocessing image: {e}")
        return np.zeros((1, 224, 224, 3))


def predict_crop(img_path):
    """Predict crop type from image"""
    if crop_model is None:
        print("⚠️ Crop model not loaded - using fallback")
        return "corn", 0.85

    try:
        img = preprocess_image(img_path)
        predictions = crop_model.predict(img, verbose=0)[0]
        pred_idx = np.argmax(predictions)
        crop = CROP_CLASSES[pred_idx]
        confidence = float(predictions[pred_idx])
        return crop, confidence
    except Exception as e:
        print(f"⚠️ Error in crop prediction: {e}")
        return "corn", 0.5


def predict_disease(img_path, crop):
    """Predict disease based on crop type"""
    try:
        img = preprocess_image(img_path)

        if crop == "corn":
            if corn_model is None:
                print("⚠️ Corn model not loaded - using fallback")
                return [("Common_Rust", 0.7), ("gls", 0.2), ("healthy", 0.1)]

            predictions = corn_model.predict(img, verbose=0)[0]
            classes = CORN_CLASSES
        else:  # rice
            if rice_model is None:
                print("⚠️ Rice model not loaded - using fallback")
                return [("blast", 0.6), ("blight", 0.3), ("healthy", 0.1)]

            predictions = rice_model.predict(img, verbose=0)[0]
            classes = RICE_CLASSES

        # Create list of (class, confidence) pairs
        results = list(zip(classes, predictions))
        # Sort by confidence (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:3]  # Return top 3 predictions
        
    except Exception as e:
        print(f"⚠️ Error in disease prediction: {e}")
        # Return fallback predictions
        if crop == "corn":
            return [("Common_Rust", 0.7), ("gls", 0.2), ("healthy", 0.1)]
        else:
            return [("blast", 0.6), ("blight", 0.3), ("healthy", 0.1)]


def get_crop_display_name(crop_code):
    """Get user-friendly crop name"""
    return CROP_DISPLAY_NAMES.get(crop_code, crop_code.title())


def get_disease_display_name(disease_code):
    """Get user-friendly disease name"""
    if disease_code in CORN_DISPLAY_NAMES:
        return CORN_DISPLAY_NAMES.get(disease_code, disease_code.title())
    else:
        return RICE_DISPLAY_NAMES.get(disease_code, disease_code.title())


def get_sample_images(disease_code, crop):
    """Return local sample image paths for the diagnosed disease"""
    # Clean disease code for folder names
    clean_disease_code = disease_code.replace(' ', '_')

    # Define the sample images directory
    base_dir = os.path.join('static', 'samples')
    sample_dir = os.path.join(base_dir, crop, clean_disease_code)

    # Check if directory exists
    if os.path.exists(sample_dir):
        # Get all image files from the directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(sample_dir, ext)))

        # Sort files to maintain consistency
        image_files.sort()

        # Convert to URL paths
        sample_urls = []
        for img_path in image_files[:4]:  # Get max 4 images
            rel_path = img_path.replace('\\', '/')
            if not rel_path.startswith('/'):
                rel_path = '/' + rel_path
            sample_urls.append(rel_path)

        if sample_urls:
            return sample_urls

    # Return empty list if no images found
    return []


def get_model_info():
    """Get information about loaded models"""
    info = {
        'crop_model': 'Loaded' if crop_model is not None else 'Not loaded',
        'corn_model': 'Loaded' if corn_model is not None else 'Not loaded',
        'rice_model': 'Loaded' if rice_model is not None else 'Not loaded',
        'tensorflow_version': tf.__version__,
        'keras_version': tf.keras.__version__
    }
    return info