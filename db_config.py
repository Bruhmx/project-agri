# db_config.py
import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from contextlib import contextmanager
import time
import ssl

load_dotenv()

# Initialize connection pool as None first
connection_pool = None

def init_db_pool():
    """Initialize the PostgreSQL connection pool with proper SSL"""
    global connection_pool
    
    try:
        # Get database URL from environment (Render provides this)
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            print("❌ DATABASE_URL not found in environment variables")
            return False
            
        print(f"🔌 Connecting to database...")
        
        # Render provides DATABASE_URL, but it might start with postgres://
        # psycopg2 requires postgresql://
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
            print("🔄 Converted postgres:// to postgresql://")
        
        # IMPORTANT: Add SSL mode to the connection string if not present
        if 'sslmode' not in database_url:
            # Add ? or & depending on whether there are already parameters
            if '?' in database_url:
                database_url += '&sslmode=require'
            else:
                database_url += '?sslmode=require'
            print("🔒 Added SSL mode: require")
        
        # Create connection pool with SSL
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1,  # min connections
            5,  # max connections (reduced for free tier)
            dsn=database_url,
            connect_timeout=10,
            sslmode='require',
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        )
        
        # Test the connection
        test_conn = connection_pool.getconn()
        test_cursor = test_conn.cursor()
        test_cursor.execute("SELECT 1")
        test_cursor.close()
        connection_pool.putconn(test_conn)
        
        print(f"✅ PostgreSQL connection pool created successfully")
        print(f"   Pool size: 1-5 connections")
        print(f"   SSL Mode: require")
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
    """Get a database connection from the pool with retry logic"""
    global connection_pool
    
    # Retry logic for cold starts and SSL issues
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if connection_pool is None:
                print(f"⚠️ Connection pool not initialized (attempt {attempt+1}/{max_retries})...")
                if not init_db_pool():
                    if attempt == max_retries - 1:
                        raise Exception("Database connection pool not initialized after retries")
                    time.sleep(retry_delay)
                    continue
            
            # Try to get a connection
            connection = connection_pool.getconn()
            
            # Test the connection is alive
            test_cursor = connection.cursor()
            test_cursor.execute("SELECT 1")
            test_cursor.close()
            
            return connection
            
        except psycopg2.OperationalError as e:
            if "SSL" in str(e):
                print(f"⚠️ SSL error (attempt {attempt+1}/{max_retries}): {e}")
                # Force reinitialize pool with fresh SSL connection
                connection_pool = None
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            else:
                print(f"⚠️ Database connection error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    
        except Exception as e:
            print(f"⚠️ Unexpected error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)
    
    raise Exception("Failed to get database connection after multiple retries")

def return_db(connection):
    """Return a connection to the pool"""
    global connection_pool
    if connection_pool and connection:
        try:
            # Ensure connection is still alive before returning
            if not connection.closed:
                connection_pool.putconn(connection)
            else:
                print("⚠️ Attempted to return closed connection to pool")
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
            try:
                connection.rollback()
            except:
                pass
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
            cursor.execute("SELECT COUNT(*) as count FROM users WHERE user_type = 'admin'")
            result = cursor.fetchone()
            admin_count = result['count'] if result else 0
            
            if admin_count == 0:
                # Create default admin user
                try:
                    from auth import hash_password
                    admin_password = hash_password('Admin@123')
                    cursor.execute("""
                        INSERT INTO users (username, email, password_hash, full_name, user_type, is_active, is_admin)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO NOTHING
                    """, ('admin', 'admin@agriaid.com', admin_password, 'System Administrator', 'admin', True, True))
                    print("  ✅ default admin user created (username: admin, password: Admin@123)")
                except Exception as e:
                    print(f"  ⚠️ Could not create admin user: {e}")
            
            print("✅✅✅ ALL TABLES CREATED/VERIFIED SUCCESSFULLY! ✅✅✅")
            
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()

# ========== AUTO-CREATE TABLES ON STARTUP ==========
print("=" * 50)
print("🚀 CHECKING DATABASE TABLES...")
print("=" * 50)
try:
    # Small delay to ensure connection pool is ready
    time.sleep(2)
    create_tables_if_not_exist()
except Exception as e:
    print(f"⚠️ Could not create tables on startup: {e}")
    print("Tables will be created when first needed.")
print("=" * 50)