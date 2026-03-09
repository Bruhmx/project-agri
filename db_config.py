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