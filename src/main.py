import os
import sys
from dotenv import load_dotenv
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load environment variables from .env and .env.local files
# .env.local will override .env
load_dotenv()
load_dotenv(dotenv_path='.env.local', override=True)

# Initialize logging system
from src.utils.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.models.user import db
from sqlalchemy import inspect, text
from src.routes.user import user_bp
from src.routes.transactions import transactions_bp
from src.routes.test_data import test_data_bp
from src.utils.validation_middleware import validation_middleware, csrf_protection

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Configurar CORS - use specific origins in production
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')
CORS(app, origins=cors_origins, supports_credentials=True, expose_headers=["Content-Type", "X-Request-ID"])

app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(transactions_bp, url_prefix='/api')
app.register_blueprint(test_data_bp, url_prefix='/api')

# Database configuration using environment variable
DATABASE_URL = os.getenv('DATABASE_URL', f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}")
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Initialize validation middleware and security features
validation_middleware.init_app(app)
csrf_protection.init_app(app)

with app.app_context():
    db.create_all()
    # Auto-migrate: ensure new columns exist in existing databases
    try:
        engine = db.engine
        inspector = inspect(engine)

        def ensure_column(table_name: str, column_name: str, column_def_sql: str):
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            if column_name not in columns:
                logger.info(f"Adding missing column '{column_name}' to table '{table_name}'")
                with engine.connect() as conn:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def_sql}"))
                    conn.commit()

        # Add justificativa columns if missing
        ensure_column('transactions', 'justificativa', 'TEXT')
        ensure_column('company_financial', 'justificativa', 'TEXT')

        # Add reconciliation_records columns if missing
        ensure_column('reconciliation_records', 'justification', 'TEXT')
        ensure_column('reconciliation_records', 'is_anomaly', 'BOOLEAN DEFAULT 0')
        ensure_column('reconciliation_records', 'anomaly_type', 'VARCHAR(50)')
        ensure_column('reconciliation_records', 'anomaly_severity', "VARCHAR(20) DEFAULT 'low'")
        ensure_column('reconciliation_records', 'anomaly_score', 'REAL')
        ensure_column('reconciliation_records', 'anomaly_reason', 'TEXT')
        ensure_column('reconciliation_records', 'anomaly_detected_at', 'DATETIME')
        ensure_column('reconciliation_records', 'anomaly_reviewed_by', 'INTEGER')
        ensure_column('reconciliation_records', 'anomaly_reviewed_at', 'DATETIME')
        ensure_column('reconciliation_records', 'anomaly_action', 'VARCHAR(20)')
        ensure_column('reconciliation_records', 'anomaly_justification', 'TEXT')
        ensure_column('reconciliation_records', 'score_breakdown', 'TEXT')
        ensure_column('reconciliation_records', 'confidence_level', "VARCHAR(20) DEFAULT 'medium'")
        ensure_column('reconciliation_records', 'risk_factors', 'TEXT')
    except Exception as e:
        logger.warning(f"Auto-migration step failed: {e}")

# Test data deletion feature controlled by environment variable
ENABLE_TEST_DATA_DELETION = os.getenv('ENABLE_TEST_DATA_DELETION', 'false').lower() == 'true'
if ENABLE_TEST_DATA_DELETION:
    # Add code here to handle test data deletion when enabled
    logger.info("Test data deletion feature is enabled")
else:
    logger.info("Test data deletion feature is disabled")

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
    # Use environment variable for debug mode, default to False for production safety
    debug_mode = os.getenv('FLASK_DEBUG', 'true').lower().strip() in ['true', '1', 'yes', 'on']
    port = int(os.getenv('FLASK_PORT', 5000))
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    
    logger.info(f"Starting Maria Conciliadora application on {host}:{port} (debug={debug_mode})")
    app.run(host=host, port=port, debug=debug_mode)
