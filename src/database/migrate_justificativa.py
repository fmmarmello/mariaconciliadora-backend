import sqlite3
import os


def add_column_if_missing(cursor, table: str, column: str, column_def: str):
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [col[1] for col in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")


def migrate_justificativa_columns():
    """
    Add 'justificativa' columns to transactions and company_financial tables if missing.
    """
    db_path = os.path.join(os.path.dirname(__file__), 'app.db')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        add_column_if_missing(cursor, 'transactions', 'justificativa', 'TEXT')
        add_column_if_missing(cursor, 'company_financial', 'justificativa', 'TEXT')

        conn.commit()
        print("Justificativa columns migration completed successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate_justificativa_columns()

