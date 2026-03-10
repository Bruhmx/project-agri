# db_helpers.py
from db_config import get_db_cursor

def insert_with_return(cursor, table, data):
    """Helper function to insert and return ID in PostgreSQL"""
    columns = data.keys()
    values = [data[col] for col in columns]
    placeholders = ['%s'] * len(columns)
    
    query = f"""
        INSERT INTO {table} 
        ({', '.join(columns)}) 
        VALUES ({', '.join(placeholders)}) 
        RETURNING id
    """
    cursor.execute(query, values)
    return cursor.fetchone()['id']

def update_diagnosis_with_answers_postgres(diagnosis_id, answers_data, summary_data):
    """PostgreSQL version of update_diagnosis_with_answers"""
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