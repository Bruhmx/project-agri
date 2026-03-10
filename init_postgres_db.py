# init_postgres_db.py
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

load_dotenv()

def init_postgres_database():
    """Initialize PostgreSQL database tables"""
    connection = None
    cursor = None
    
    try:
        # Get database URL
        database_url = os.getenv('DATABASE_URL')
        if database_url and database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
        if not database_url:
            print("❌ DATABASE_URL not found in environment variables")
            return False
        
        print(f"✅ Connecting to database...")
        
        # Connect to PostgreSQL
        connection = psycopg2.connect(database_url)
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()
        
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
        print("✅ Created users table")
        
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
        print("✅ Created diagnosis_history table")
        
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
        print("✅ Created disease_info table")
        
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
        print("✅ Created disease_samples table")
        
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
        print("✅ Created questions table")
        
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
        print("✅ Created user_settings table")
        
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
        print("✅ Created saved_diagnoses table")
        
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
        print("✅ Created feedback table")
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_user_id ON diagnosis_history(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_crop ON diagnosis_history(crop)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_disease ON diagnosis_history(disease_detected)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_status ON diagnosis_history(expert_review_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_crop_disease ON questions(crop, disease_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)")
        
        print("✅ Created indexes")
        
        # Check if admin user exists, if not create default admin
        cursor.execute("SELECT COUNT(*) FROM users WHERE user_type = 'admin'")
        admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            # Create default admin user (you should change these credentials)
            from auth import hash_password
            admin_password = hash_password('Admin@123')
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, full_name, user_type, is_active, is_admin)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, ('admin', 'admin@agriaid.com', admin_password, 'System Administrator', 'admin', True, True))
            print("✅ Created default admin user (username: admin, password: Admin@123)")
        
        cursor.close()
        connection.close()
        print("\n✅✅✅ DATABASE INITIALIZATION COMPLETE! ✅✅✅")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    init_postgres_database()