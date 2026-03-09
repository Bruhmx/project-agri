# db_config.py
import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from contextlib import contextmanager
import time

load_dotenv()

# Initialize connection pool as None first
connection_pool = None

def init_db_pool():
    """Initialize the PostgreSQL connection pool"""
    global connection_pool
    
    try:
        # Get database URL from environment (Render provides this)
        database_url = os.getenv('DATABASE_URL')
        
        # Render provides DATABASE_URL, but it might start with postgres://
        # psycopg2 requires postgresql://
        if database_url and database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        if database_url:
            # Use the connection string directly
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                1,  # min connections
                5,  # max connections (reduced for free tier)
                dsn=database_url
            )
            print(f"✅ PostgreSQL connection pool created using DATABASE_URL")
        else:
            # Fallback to individual parameters
            db_config = {
                "host": os.getenv("DB_HOST", "localhost"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", ""),
                "database": os.getenv("DB_NAME", "agriaid"),
                "port": int(os.getenv("DB_PORT", 5432)),
            }
            
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 5, **db_config
            )
            print(f"✅ PostgreSQL connection pool created with parameters")
        
        print(f"   Pool size: 1-5 connections")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create connection pool: {e}")
        import traceback
        traceback.print_exc()
        connection_pool = None
        return False

# Initialize the pool when module is imported
init_db_pool()

def get_db():
    """Get a database connection from the pool"""
    global connection_pool
    
    # Retry logic for cold starts
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if connection_pool is None:
                print("⚠️ Connection pool not initialized, attempting to reinitialize...")
                if not init_db_pool():
                    if attempt == max_retries - 1:
                        raise Exception("Database connection pool not initialized after retries")
                    time.sleep(2)
                    continue
            
            connection = connection_pool.getconn()
            return connection
        except Exception as e:
            print(f"⚠️ Error getting database connection (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
    
    raise Exception("Failed to get database connection")

def return_db(connection):
    """Return a connection to the pool"""
    global connection_pool
    if connection_pool and connection:
        try:
            connection_pool.putconn(connection)
        except Exception as e:
            print(f"⚠️ Error returning connection to pool: {e}")

@contextmanager
def get_db_cursor():
    """Context manager for database connections with commit"""
    connection = None
    cursor = None
    try:
        connection = get_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        yield cursor
        connection.commit()
    except Exception as e:
        if connection:
            connection.rollback()
        raise e
    finally:
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if connection:
            try:
                return_db(connection)
            except:
                pass

@contextmanager
def get_db_cursor_readonly():
    """Context manager for read-only operations"""
    connection = None
    cursor = None
    try:
        connection = get_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        yield cursor
    finally:
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if connection:
            try:
                return_db(connection)
            except:
                pass

def get_pool_info():
    """Get information about the connection pool"""
    global connection_pool
    
    if connection_pool is None:
        return {"status": "not_initialized"}
    
    try:
        return {
            "status": "active",
            "min_connections": connection_pool.minconn,
            "max_connections": connection_pool.maxconn,
            "closed": getattr(connection_pool, '_closed', False)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ========== TABLE CREATION FUNCTION ==========

def create_tables_if_not_exist():
    """Create all necessary tables if they don't exist"""
    print("📦 Checking/Creating database tables...")
    try:
        with get_db_cursor() as cursor:
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(100),
                    phone_number VARCHAR(20),
                    location VARCHAR(255),
                    language VARCHAR(10) DEFAULT 'en',
                    bio TEXT,
                    profile_image VARCHAR(255),
                    user_type VARCHAR(20) DEFAULT 'farmer',
                    is_active BOOLEAN DEFAULT TRUE,
                    is_admin BOOLEAN DEFAULT FALSE,
                    last_login TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("  ✅ users table ready")
            
            # Create diagnosis_history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS diagnosis_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    image_path VARCHAR(255),
                    crop VARCHAR(50),
                    disease_detected VARCHAR(100),
                    confidence FLOAT,
                    symptoms TEXT,
                    recommendations TEXT,
                    location VARCHAR(255),
                    expert_answers JSONB,
                    expert_summary JSONB,
                    final_confidence_level VARCHAR(50),
                    for_training BOOLEAN DEFAULT TRUE,
                    training_used BOOLEAN DEFAULT FALSE,
                    image_processed BOOLEAN DEFAULT FALSE,
                    expert_review_status VARCHAR(20) DEFAULT 'pending',
                    reviewed_by INTEGER REFERENCES users(id),
                    reviewed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("  ✅ diagnosis_history table ready")
            
            # Create disease_info table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS disease_info (
                    id SERIAL PRIMARY KEY,
                    crop VARCHAR(50),
                    disease_code VARCHAR(50),
                    disease_name VARCHAR(100),
                    cause TEXT,
                    symptoms TEXT,
                    organic_treatment TEXT,
                    chemical_treatment TEXT,
                    manual_treatment TEXT,
                    prevention TEXT,
                    status VARCHAR(20) DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(crop, disease_code)
                )
            """)
            print("  ✅ disease_info table ready")
            
            # Create disease_samples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS disease_samples (
                    id SERIAL PRIMARY KEY,
                    crop VARCHAR(50),
                    disease_code VARCHAR(50),
                    image_path VARCHAR(255),
                    image_title VARCHAR(255),
                    image_description TEXT,
                    severity_level VARCHAR(50),
                    display_order INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("  ✅ disease_samples table ready")
            
            # Create questions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id SERIAL PRIMARY KEY,
                    crop VARCHAR(50),
                    disease_code VARCHAR(50),
                    question_text TEXT,
                    yes_score INTEGER DEFAULT 5,
                    no_score INTEGER DEFAULT 0,
                    question_category VARCHAR(50),
                    priority INTEGER DEFAULT 1,
                    depends_on INTEGER REFERENCES questions(id),
                    show_if_answer VARCHAR(10),
                    display_order INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("  ✅ questions table ready")
            
            # Create user_settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    two_factor_enabled BOOLEAN DEFAULT FALSE,
                    email_notifications BOOLEAN DEFAULT TRUE,
                    dark_mode BOOLEAN DEFAULT FALSE,
                    email_updates BOOLEAN DEFAULT TRUE,
                    email_newsletter BOOLEAN DEFAULT FALSE,
                    email_promotions BOOLEAN DEFAULT FALSE,
                    app_notifications BOOLEAN DEFAULT TRUE,
                    app_security BOOLEAN DEFAULT TRUE,
                    app_reminders BOOLEAN DEFAULT TRUE,
                    frequency VARCHAR(20) DEFAULT 'realtime',
                    profile_public BOOLEAN DEFAULT TRUE,
                    show_diagnosis BOOLEAN DEFAULT TRUE,
                    data_collection BOOLEAN DEFAULT TRUE,
                    theme VARCHAR(20) DEFAULT 'light',
                    density VARCHAR(20) DEFAULT 'comfortable',
                    auto_save BOOLEAN DEFAULT TRUE,
                    show_tips BOOLEAN DEFAULT TRUE,
                    detailed_results BOOLEAN DEFAULT TRUE,
                    quick_analysis BOOLEAN DEFAULT TRUE,
                    default_crop VARCHAR(20),
                    measurement_unit VARCHAR(20) DEFAULT 'metric',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id)
                )
            """)
            print("  ✅ user_settings table ready")
            
            # Create saved_diagnoses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS saved_diagnoses (
                    id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    crop VARCHAR(50),
                    disease VARCHAR(100),
                    confidence FLOAT,
                    symptoms TEXT,
                    recommendations TEXT,
                    status VARCHAR(50),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, id)
                )
            """)
            print("  ✅ saved_diagnoses table ready")
            
            # Create feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    diagnosis_id INTEGER REFERENCES diagnosis_history(id) ON DELETE SET NULL,
                    name VARCHAR(100),
                    email VARCHAR(100),
                    feedback_type VARCHAR(50),
                    subject VARCHAR(255),
                    message TEXT,
                    image_path VARCHAR(255),
                    rating INTEGER,
                    accuracy_rating INTEGER,
                    feedback_text TEXT,
                    suggestions TEXT,
                    admin_response TEXT,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("  ✅ feedback table ready")
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_user_id ON diagnosis_history(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_crop ON diagnosis_history(crop)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_disease ON diagnosis_history(disease_detected)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_status ON diagnosis_history(expert_review_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_crop_disease ON questions(crop, disease_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)")
            
            print("  ✅ indexes created")
            
            # Check if admin user exists, if not create default admin
            cursor.execute("SELECT COUNT(*) FROM users WHERE user_type = 'admin'")
            admin_count = cursor.fetchone()['count']
            
            if admin_count == 0:
                # Create default admin user
                from auth import hash_password
                admin_password = hash_password('Admin@123')
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, full_name, user_type, is_active, is_admin)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (username) DO NOTHING
                """, ('admin', 'admin@agriaid.com', admin_password, 'System Administrator', 'admin', True, True))
                print("  ✅ default admin user created (username: admin, password: Admin@123)")
            
            print("✅✅✅ ALL TABLES CREATED/VERIFIED SUCCESSFULLY! ✅✅✅")
            
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()

# ========== AUTO-CREATE TABLES ON STARTUP ==========
# This will run every time the app starts
print("=" * 50)
print("🚀 CHECKING DATABASE TABLES...")
print("=" * 50)
try:
    create_tables_if_not_exist()
except Exception as e:
    print(f"⚠️ Could not create tables on startup: {e}")
    print("Tables will be created when first needed.")
print("=" * 50)