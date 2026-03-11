# db_config.py
import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from contextlib import contextmanager
import time
import threading
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize connection pool as None first
connection_pool = None
_pool_lock = threading.Lock()
_pool_initialized = False

# Configuration
POOL_MIN_CONN = 1
POOL_MAX_CONN = 10  # Increased slightly for better handling
CONNECTION_TIMEOUT = 30
RETRY_DELAY = 1
MAX_RETRIES = 3

def init_db_pool():
    """Initialize the PostgreSQL connection pool with proper SSL"""
    global connection_pool, _pool_initialized
    
    # Use lock to prevent multiple threads from initializing simultaneously
    with _pool_lock:
        if _pool_initialized and connection_pool is not None:
            return True
            
        try:
            # Get database URL from environment (Render provides this)
            database_url = os.getenv('DATABASE_URL')
            
            if not database_url:
                logger.error("❌ DATABASE_URL not found in environment variables")
                return False
                
            logger.info("🔌 Connecting to database...")
            
            # Render provides DATABASE_URL, but it might start with postgres://
            # psycopg2 requires postgresql://
            if database_url.startswith('postgres://'):
                database_url = database_url.replace('postgres://', 'postgresql://', 1)
                logger.info("🔄 Converted postgres:// to postgresql://")
            
            # IMPORTANT: Add SSL mode to the connection string if not present
            if 'sslmode' not in database_url:
                # Add ? or & depending on whether there are already parameters
                if '?' in database_url:
                    database_url += '&sslmode=require'
                else:
                    database_url += '?sslmode=require'
                logger.info("🔒 Added SSL mode: require")
            
            # Create connection pool with SSL
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                POOL_MIN_CONN,
                POOL_MAX_CONN,
                dsn=database_url,
                connect_timeout=10,
                sslmode='require',
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            
            # Test the connection
            test_conn = None
            try:
                test_conn = connection_pool.getconn()
                with test_conn.cursor() as test_cursor:
                    test_cursor.execute("SELECT 1")
                logger.info("✅ Connection test successful")
            finally:
                if test_conn:
                    connection_pool.putconn(test_conn)
            
            _pool_initialized = True
            logger.info(f"✅ PostgreSQL connection pool created successfully")
            logger.info(f"   Pool size: {POOL_MIN_CONN}-{POOL_MAX_CONN} connections")
            logger.info(f"   SSL Mode: require")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create connection pool: {e}")
            import traceback
            traceback.print_exc()
            connection_pool = None
            _pool_initialized = False
            return False

# Initialize the pool when module is imported
init_db_pool()

def get_db():
    """Get a database connection from the pool with retry logic"""
    global connection_pool, _pool_initialized
    
    # Retry logic for cold starts and SSL issues
    for attempt in range(MAX_RETRIES):
        try:
            # Check if pool needs initialization
            if connection_pool is None or not _pool_initialized:
                logger.warning(f"⚠️ Connection pool not initialized (attempt {attempt+1}/{MAX_RETRIES})...")
                if not init_db_pool():
                    if attempt == MAX_RETRIES - 1:
                        raise Exception("Database connection pool not initialized after retries")
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    continue
            
            # Try to get a connection with timeout
            try:
                connection = connection_pool.getconn()
            except Exception as e:
                logger.error(f"Failed to get connection from pool: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            
            # Test the connection is alive
            try:
                with connection.cursor() as test_cursor:
                    test_cursor.execute("SELECT 1")
                return connection
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                # Connection is dead, close it and try again
                logger.warning(f"Connection test failed: {e}")
                try:
                    connection_pool.putconn(connection, close=True)
                except:
                    pass
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY * (attempt + 1))
                
        except psycopg2.OperationalError as e:
            if "SSL" in str(e):
                logger.warning(f"⚠️ SSL error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                # Force reinitialize pool with fresh SSL connection
                with _pool_lock:
                    connection_pool = None
                    _pool_initialized = False
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.warning(f"⚠️ Database connection error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    raise Exception("Failed to get database connection after multiple retries")

def return_db(connection):
    """Return a connection to the pool safely"""
    global connection_pool
    if connection_pool is None or connection is None:
        return
    
    try:
        # Check if connection is still alive before returning
        if not connection.closed:
            try:
                # Quick test to ensure connection is usable
                with connection.cursor() as test_cursor:
                    test_cursor.execute("SELECT 1")
                connection_pool.putconn(connection)
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                # Connection is dead, close it properly
                logger.warning("Returning dead connection to pool, closing instead")
                try:
                    connection_pool.putconn(connection, close=True)
                except:
                    try:
                        connection.close()
                    except:
                        pass
        else:
            logger.warning("Attempted to return closed connection to pool")
            # Connection already closed, just log it
            pass
    except Exception as e:
        logger.error(f"⚠️ Error returning connection to pool: {e}")
        # Try to close the connection manually
        try:
            connection.close()
        except:
            pass

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
        logger.error(f"❌ Database error: {e}")
        raise
    finally:
        # Always clean up in reverse order
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if connection:
            return_db(connection)

@contextmanager
def get_db_cursor_readonly():
    """Context manager for read-only operations"""
    connection = None
    cursor = None
    try:
        connection = get_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        # Set transaction to readonly for safety
        cursor.execute("SET TRANSACTION READ ONLY")
        yield cursor
    except Exception as e:
        logger.error(f"❌ Readonly database error: {e}")
        raise
    finally:
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if connection:
            return_db(connection)

def get_pool_info():
    """Get information about the connection pool"""
    global connection_pool, _pool_initialized
    
    if connection_pool is None:
        return {
            "status": "not_initialized",
            "initialized": _pool_initialized
        }
    
    try:
        # Try to get pool stats (may not be available on all versions)
        used = getattr(connection_pool, '_used', 'N/A')
        pool_size = getattr(connection_pool, '_pool', 'N/A')
        
        return {
            "status": "active",
            "initialized": _pool_initialized,
            "min_connections": POOL_MIN_CONN,
            "max_connections": POOL_MAX_CONN,
            "used_connections": len(used) if isinstance(used, (list, dict)) else 'N/A',
            "pool_size": len(pool_size) if isinstance(pool_size, (list, dict)) else 'N/A',
            "closed": getattr(connection_pool, '_closed', False)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "initialized": _pool_initialized
        }

def close_all_connections():
    """Close all connections in the pool - call this on app shutdown"""
    global connection_pool, _pool_initialized
    with _pool_lock:
        if connection_pool:
            try:
                connection_pool.closeall()
                logger.info("✅ All database connections closed")
            except Exception as e:
                logger.error(f"❌ Error closing connections: {e}")
            finally:
                connection_pool = None
                _pool_initialized = False

# ========== TABLE CREATION FUNCTION ==========

def create_tables_if_not_exist():
    """Create all necessary tables if they don't exist"""
    logger.info("📦 Checking/Creating database tables...")
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
            logger.info("  ✅ users table ready")
            
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
            logger.info("  ✅ diagnosis_history table ready")
            
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
            logger.info("  ✅ disease_info table ready")
            
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
            logger.info("  ✅ disease_samples table ready")
            
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
            logger.info("  ✅ questions table ready")
            
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
            logger.info("  ✅ user_settings table ready")
            
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
            logger.info("  ✅ saved_diagnoses table ready")
            
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
            logger.info("  ✅ feedback table ready")
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_user_id ON diagnosis_history(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_crop ON diagnosis_history(crop)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_disease ON diagnosis_history(disease_detected)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_status ON diagnosis_history(expert_review_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_crop_disease ON questions(crop, disease_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)")
            
            logger.info("  ✅ indexes created")
            
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
                    logger.info("  ✅ default admin user created (username: admin, password: Admin@123)")
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not create admin user: {e}")
            
            logger.info("✅✅✅ ALL TABLES CREATED/VERIFIED SUCCESSFULLY! ✅✅✅")
            
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()

# ========== AUTO-CREATE TABLES ON STARTUP ==========
logger.info("=" * 50)
logger.info("🚀 CHECKING DATABASE TABLES...")
logger.info("=" * 50)
try:
    # Small delay to ensure connection pool is ready
    time.sleep(2)
    create_tables_if_not_exist()
except Exception as e:
    logger.warning(f"⚠️ Could not create tables on startup: {e}")
    logger.warning("Tables will be created when first needed.")
logger.info("=" * 50)

# Add cleanup handler
import atexit
atexit.register(close_all_connections)