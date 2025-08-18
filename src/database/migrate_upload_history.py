import sqlite3
import os

def migrate_upload_history_table():
    """
    Add missing columns to the upload_history table
    """
    db_path = os.path.join(os.path.dirname(__file__), 'app.db')
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the columns already exist
    cursor.execute("PRAGMA table_info(upload_history)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Add missing columns if they don't exist
    if 'file_hash' not in columns:
        cursor.execute("ALTER TABLE upload_history ADD COLUMN file_hash VARCHAR(64)")
    
    if 'duplicate_files_count' not in columns:
        cursor.execute("ALTER TABLE upload_history ADD COLUMN duplicate_files_count INTEGER DEFAULT 0")
    
    if 'duplicate_entries_count' not in columns:
        cursor.execute("ALTER TABLE upload_history ADD COLUMN duplicate_entries_count INTEGER DEFAULT 0")
    
    if 'total_entries_processed' not in columns:
        cursor.execute("ALTER TABLE upload_history ADD COLUMN total_entries_processed INTEGER DEFAULT 0")
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("UploadHistory table migration completed successfully.")

if __name__ == "__main__":
    migrate_upload_history_table()