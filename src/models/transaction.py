from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from src.models.user import db

class Transaction(db.Model):
    __tablename__ = 'transactions'
    
    id = db.Column(db.Integer, primary_key=True)
    bank_name = db.Column(db.String(50), nullable=False)
    account_id = db.Column(db.String(100), nullable=False)
    transaction_id = db.Column(db.String(100), nullable=True)
    date = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=False)
    transaction_type = db.Column(db.String(20), nullable=False)  # 'debit' or 'credit'
    balance = db.Column(db.Float, nullable=True)
    category = db.Column(db.String(50), nullable=True)  # Categoria sugerida pela IA
    is_anomaly = db.Column(db.Boolean, default=False)  # Marcada como anomalia pela IA
    justificativa = db.Column(db.Text, nullable=True)  # Justificativa de ajuste manual
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'bank_name': self.bank_name,
            'account_id': self.account_id,
            'transaction_id': self.transaction_id,
            'date': self.date.isoformat() if self.date else None,
            'amount': self.amount,
            'description': self.description,
            'transaction_type': self.transaction_type,
            'balance': self.balance,
            'category': self.category,
            'is_anomaly': self.is_anomaly,
            'justificativa': self.justificativa,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ReconciliationRecord(db.Model):
    __tablename__ = 'reconciliation_records'
    
    id = db.Column(db.Integer, primary_key=True)
    bank_transaction_id = db.Column(db.Integer, db.ForeignKey('transactions.id'), nullable=False)
    company_entry_id = db.Column(db.Integer, db.ForeignKey('company_financial.id'), nullable=False)
    match_score = db.Column(db.Float, nullable=False)  # Score de 0 a 1 indicando a qualidade do match
    status = db.Column(db.String(20), default='pending')  # 'pending', 'confirmed', 'rejected'
    justification = db.Column(db.Text, nullable=True)  # Justification for manual matches or adjustments
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Anomaly Detection Fields
    is_anomaly = db.Column(db.Boolean, default=False)  # Flagged as potential anomaly
    anomaly_type = db.Column(db.String(50), nullable=True)  # Type of anomaly detected
    anomaly_severity = db.Column(db.String(20), default='low')  # 'low', 'medium', 'high', 'critical'
    anomaly_score = db.Column(db.Float, nullable=True)  # Anomaly confidence score (0-1)
    anomaly_reason = db.Column(db.Text, nullable=True)  # Detailed explanation of anomaly
    anomaly_detected_at = db.Column(db.DateTime, nullable=True)  # When anomaly was detected
    anomaly_reviewed_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # User who reviewed
    anomaly_reviewed_at = db.Column(db.DateTime, nullable=True)  # When anomaly was reviewed
    anomaly_action = db.Column(db.String(20), nullable=True)  # 'confirmed', 'dismissed', 'escalated'
    anomaly_justification = db.Column(db.Text, nullable=True)  # User justification for action
    
    # Enhanced matching fields
    score_breakdown = db.Column(db.Text, nullable=True)  # JSON string with detailed score components
    confidence_level = db.Column(db.String(20), default='medium')  # 'low', 'medium', 'high'
    risk_factors = db.Column(db.Text, nullable=True)  # JSON string with identified risk factors
    
    # Relacionamentos
    bank_transaction = db.relationship('Transaction', foreign_keys=[bank_transaction_id])
    company_entry = db.relationship('CompanyFinancial', foreign_keys=[company_entry_id])
    reviewer = db.relationship('User', foreign_keys=[anomaly_reviewed_by])
    
    def to_dict(self):
        return {
            'id': self.id,
            'bank_transaction_id': self.bank_transaction_id,
            'company_entry_id': self.company_entry_id,
            'match_score': self.match_score,
            'status': self.status,
            'justification': self.justification,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            
            # Anomaly Detection Fields
            'is_anomaly': self.is_anomaly,
            'anomaly_type': self.anomaly_type,
            'anomaly_severity': self.anomaly_severity,
            'anomaly_score': self.anomaly_score,
            'anomaly_reason': self.anomaly_reason,
            'anomaly_detected_at': self.anomaly_detected_at.isoformat() if self.anomaly_detected_at else None,
            'anomaly_reviewed_by': self.anomaly_reviewed_by,
            'anomaly_reviewed_at': self.anomaly_reviewed_at.isoformat() if self.anomaly_reviewed_at else None,
            'anomaly_action': self.anomaly_action,
            'anomaly_justification': self.anomaly_justification,
            
            # Enhanced matching fields
            'score_breakdown': self.get_score_breakdown(),
            'confidence_level': self.confidence_level,
            'risk_factors': self.get_risk_factors(),
            
            'bank_transaction': self.bank_transaction.to_dict() if self.bank_transaction else None,
            'company_entry': self.company_entry.to_dict() if self.company_entry else None
        }
    
    def get_score_breakdown(self):
        """Parse JSON score breakdown"""
        try:
            import json
            return json.loads(self.score_breakdown) if self.score_breakdown else {}
        except:
            return {}
    
    def get_risk_factors(self):
        """Parse JSON risk factors"""
        try:
            import json
            return json.loads(self.risk_factors) if self.risk_factors else {}
        except:
            return {}
    
    def flag_anomaly(self, anomaly_type, severity, score, reason, user_id=None):
        """Flag this reconciliation record as an anomaly"""
        from datetime import datetime
        import json
        
        self.is_anomaly = True
        self.anomaly_type = anomaly_type
        self.anomaly_severity = severity
        self.anomaly_score = score
        self.anomaly_reason = reason
        self.anomaly_detected_at = datetime.utcnow()
        self.anomaly_action = 'pending' if user_id else None
        
        if user_id:
            self.anomaly_reviewed_by = user_id
            self.anomaly_reviewed_at = datetime.utcnow()
    
    def resolve_anomaly(self, action, justification, user_id):
        """Resolve an anomaly with user action"""
        from datetime import datetime
        
        self.anomaly_action = action
        self.anomaly_justification = justification
        self.anomaly_reviewed_by = user_id
        self.anomaly_reviewed_at = datetime.utcnow()
        
        # Update status based on action
        if action == 'confirmed':
            self.status = 'confirmed'
        elif action == 'dismissed':
            self.is_anomaly = False
class UploadHistory(db.Model):
    __tablename__ = 'upload_history'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    bank_name = db.Column(db.String(50), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    transactions_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='success')  # 'success', 'error', 'partial'
    error_message = db.Column(db.Text, nullable=True)
    
    # New fields for duplicate tracking
    file_hash = db.Column(db.String(64), nullable=True)  # SHA-256 hash of the file
    duplicate_files_count = db.Column(db.Integer, default=0)  # Number of duplicate files detected
    duplicate_entries_count = db.Column(db.Integer, default=0)  # Number of duplicate entries detected
    total_entries_processed = db.Column(db.Integer, default=0)  # Total entries processed (including duplicates)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'bank_name': self.bank_name,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'transactions_count': self.transactions_count,
            'status': self.status,
            'error_message': self.error_message,
            'file_hash': self.file_hash,
            'duplicate_files_count': self.duplicate_files_count,
            'duplicate_entries_count': self.duplicate_entries_count,
            'total_entries_processed': self.total_entries_processed
        }

