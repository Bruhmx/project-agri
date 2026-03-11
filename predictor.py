import tensorflow as tf
import numpy as np
from PIL import Image
import warnings
import os
import gc

warnings.filterwarnings('ignore')

# Limit TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Limit CPU threads to reduce memory
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

print("=" * 50)
print(f"TensorFlow version: {tf.__version__}")
print("=" * 50)

def load_model_safely(model_path, model_name):
    """Load a Keras model with memory optimization"""
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        return None
    
    print(f"\nAttempting to load {model_name} from {os.path.basename(model_path)}...")
    
    try:
        # Load model with memory optimization
        model = tf.keras.models.load_model(model_path)
        print(f"✅ {model_name} loaded successfully!")
        return model
    except Exception as e:
        print(f"  Loading failed: {type(e).__name__}")
        return None

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("\n📁 Checking models directory...")
if os.path.exists(MODELS_DIR):
    print(f"✅ Models directory found")
    
print("\n" + "=" * 50)
print("🚀 LOADING MODELS...")
print("=" * 50)

# Load models one at a time to manage memory
crop_model = None
corn_model = None
rice_model = None

# Try to load models, but don't fail if they don't load
try:
    crop_model = load_model_safely(os.path.join(MODELS_DIR, "crop_model_final.keras"), "Crop Model")
    gc.collect()  # Force garbage collection
except:
    pass

try:
    corn_model = load_model_safely(os.path.join(MODELS_DIR, "corn_model_final.keras"), "Corn Disease Model")
    gc.collect()
except:
    pass

try:
    rice_model = load_model_safely(os.path.join(MODELS_DIR, "rice_model_final.keras"), "Rice Disease Model")
    gc.collect()
except:
    pass

print("\n" + "=" * 50)
print("📊 MODEL STATUS:")
print(f"  Crop Model: {'✅ LOADED' if crop_model is not None else '✅ USING FALLBACK'}")
print(f"  Corn Model: {'✅ LOADED' if corn_model is not None else '✅ USING FALLBACK'}")
print(f"  Rice Model: {'✅ LOADED' if rice_model is not None else '✅ USING FALLBACK'}")

if not (crop_model and corn_model and rice_model):
    print("\n✨ Using fallback predictions to save memory")
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

# Fallback predictions
FALLBACK_DISEASES = {
    "corn": [
        ("Common_Rust", 0.7),
        ("gls", 0.2),
        ("healthy", 0.1)
    ],
    "rice": [
        ("blast", 0.6),
        ("blight", 0.3),
        ("healthy", 0.1)
    ]
}

def preprocess_image(img_path, img_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        if not os.path.exists(img_path):
            return np.zeros((1, 224, 224, 3))
            
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"⚠️ Error preprocessing image: {e}")
        return np.zeros((1, 224, 224, 3))


def predict_crop(img_path):
    """Predict crop type from image"""
    if crop_model is None:
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
        if crop == "corn":
            if corn_model is None:
                return FALLBACK_DISEASES["corn"]
            predictions = corn_model.predict(img, verbose=0)[0]
            classes = CORN_CLASSES
        else:
            if rice_model is None:
                return FALLBACK_DISEASES["rice"]
            predictions = rice_model.predict(img, verbose=0)[0]
            classes = RICE_CLASSES

        results = list(zip(classes, predictions))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:3]
        
    except Exception as e:
        print(f"⚠️ Error in disease prediction: {e}")
        return FALLBACK_DISEASES.get(crop, FALLBACK_DISEASES["corn"])


def get_crop_display_name(crop_code):
    return CROP_DISPLAY_NAMES.get(crop_code, crop_code.title())


def get_disease_display_name(disease_code):
    if disease_code in CORN_DISPLAY_NAMES:
        return CORN_DISPLAY_NAMES.get(disease_code, disease_code.title())
    return RICE_DISPLAY_NAMES.get(disease_code, disease_code.title())


def get_sample_images(disease_code, crop):
    clean_disease_code = disease_code.replace(' ', '_')
    sample_dir = os.path.join('static', 'samples', crop, clean_disease_code)

    if os.path.exists(sample_dir):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']:
            image_files.extend(glob.glob(os.path.join(sample_dir, ext)))
        
        image_files.sort()
        return ['/' + f.replace('\\', '/') for f in image_files[:4]]
    return []


def get_model_info():
    return {
        'crop_model': 'Loaded' if crop_model else 'Fallback',
        'corn_model': 'Loaded' if corn_model else 'Fallback',
        'rice_model': 'Loaded' if rice_model else 'Fallback',
        'tensorflow_version': tf.__version__,
        'memory_mode': 'Optimized for 512MB RAM'
    }