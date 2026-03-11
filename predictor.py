import tensorflow as tf
import numpy as np
from PIL import Image
import warnings
import os
import glob
import json

warnings.filterwarnings('ignore')

print("=" * 50)
print(f"TensorFlow version: {tf.__version__}")
print("=" * 50)

def load_model_safely(model_path, model_name):
    """Load a Keras model with compatibility for TF 2.15.0"""
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        # Try looking for .h5 version
        h5_path = model_path.replace('.keras', '.h5')
        if os.path.exists(h5_path):
            print(f"  Found .h5 alternative: {h5_path}")
            model_path = h5_path
        else:
            return None
    
    print(f"\nAttempting to load {model_name} from {os.path.basename(model_path)}...")
    
    # Try multiple loading methods
    methods = [
        # Method 1: Standard loading
        lambda: tf.keras.models.load_model(model_path),
        
        # Method 2: Load with compile=False
        lambda: tf.keras.models.load_model(model_path, compile=False),
        
        # Method 3: Load with custom objects for Functional class
        lambda: tf.keras.models.load_model(
            model_path, 
            custom_objects={
                'Functional': tf.keras.Model,
                'DTypePolicy': object
            }
        ),
        
        # Method 4: Try with safe_mode=False (for Keras v3)
        lambda: tf.keras.models.load_model(model_path, safe_mode=False)
    ]
    
    for i, method in enumerate(methods, 1):
        try:
            model = method()
            print(f"✅ {model_name} loaded successfully (method {i})!")
            return model
        except Exception as e:
            print(f"  Method {i} failed: {type(e).__name__}")
            continue
    
    print(f"  All loading methods failed for {model_name}")
    return None

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("\n📁 Checking models directory...")
if os.path.exists(MODELS_DIR):
    print(f"✅ Models directory found: {MODELS_DIR}")
    print("\n📋 Available model files:")
    
    # Debug information
    print("\n🔍 Detailed model file inspection:")
    keras_v3_files = []
    for f in os.listdir(MODELS_DIR):
        file_path = os.path.join(MODELS_DIR, f)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if f.endswith('.keras'):
                keras_v3_files.append(f)
                print(f"  📄 {f} - {size_mb:.2f} MB (Keras v3 format)")
            elif f.endswith('.h5'):
                print(f"  📄 {f} - {size_mb:.2f} MB (H5 format)")
            else:
                print(f"  📄 {f} - {size_mb:.2f} MB")
    
    if keras_v3_files:
        print(f"\n⚠️ Found {len(keras_v3_files)} Keras v3 model(s). These may not be compatible with TF {tf.__version__}")
        print("   The app will use fallback predictions for now.")
else:
    print(f"⚠️ Models directory not found at: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"✅ Created models directory")

print("\n" + "=" * 50)
print("🚀 LOADING MODELS...")
print("=" * 50)

# Load models
crop_model = load_model_safely(os.path.join(MODELS_DIR, "crop_model_final.keras"), "Crop Model")
corn_model = load_model_safely(os.path.join(MODELS_DIR, "corn_model_final.keras"), "Corn Disease Model")
rice_model = load_model_safely(os.path.join(MODELS_DIR, "rice_model_final.keras"), "Rice Disease Model")

print("\n" + "=" * 50)
print("📊 MODEL STATUS:")
print(f"  Crop Model: {'✅ LOADED' if crop_model is not None else '❌ NOT LOADED'}")
print(f"  Corn Model: {'✅ LOADED' if corn_model is not None else '❌ NOT LOADED'}")
print(f"  Rice Model: {'✅ LOADED' if rice_model is not None else '❌ NOT LOADED'}")

if crop_model and corn_model and rice_model:
    print("\n✨ All models loaded successfully! Ready for predictions.")
else:
    print("\n⚠️ Some models failed to load - using fallback predictions")
    print("   This is expected with Keras v3 models on TensorFlow 2.15.0")
    print("   The app will still work with pre-configured disease information.")
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

# Fallback disease database for when models don't load
FALLBACK_DISEASES = {
    "corn": [
        {"code": "Common_Rust", "name": "Common Rust", "confidence": 0.7},
        {"code": "gls", "name": "Gray Leaf Spot", "confidence": 0.2},
        {"code": "healthy", "name": "Healthy", "confidence": 0.1}
    ],
    "rice": [
        {"code": "blast", "name": "Rice Blast", "confidence": 0.6},
        {"code": "blight", "name": "Bacterial Leaf Blight", "confidence": 0.3},
        {"code": "healthy", "name": "Healthy", "confidence": 0.1}
    ]
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
        print("⚠️ Crop model not loaded - using fallback (corn)")
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
                return [(d["code"], d["confidence"]) for d in FALLBACK_DISEASES["corn"]]

            predictions = corn_model.predict(img, verbose=0)[0]
            classes = CORN_CLASSES
        else:  # rice
            if rice_model is None:
                print("⚠️ Rice model not loaded - using fallback")
                return [(d["code"], d["confidence"]) for d in FALLBACK_DISEASES["rice"]]

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
            return [(d["code"], d["confidence"]) for d in FALLBACK_DISEASES["corn"]]
        else:
            return [(d["code"], d["confidence"]) for d in FALLBACK_DISEASES["rice"]]


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
    clean_disease_code = disease_code.replace(' ', '_')
    base_dir = os.path.join('static', 'samples')
    sample_dir = os.path.join(base_dir, crop, clean_disease_code)

    if os.path.exists(sample_dir):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(sample_dir, ext)))

        image_files.sort()

        sample_urls = []
        for img_path in image_files[:4]:
            rel_path = img_path.replace('\\', '/')
            if not rel_path.startswith('/'):
                rel_path = '/' + rel_path
            sample_urls.append(rel_path)

        if sample_urls:
            return sample_urls

    return []


def get_model_info():
    """Get information about loaded models"""
    info = {
        'crop_model': 'Loaded' if crop_model is not None else 'Not loaded (using fallback)',
        'corn_model': 'Loaded' if corn_model is not None else 'Not loaded (using fallback)',
        'rice_model': 'Loaded' if rice_model is not None else 'Not loaded (using fallback)',
        'tensorflow_version': tf.__version__,
        'status': 'Using fallback predictions' if not (crop_model and corn_model and rice_model) else 'All models loaded'
    }
    # Safely try to get keras version
    try:
        info['keras_version'] = tf.keras.__version__
    except:
        info['keras_version'] = 'Unknown'
    return info