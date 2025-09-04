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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relacionamentos
    bank_transaction = db.relationship('Transaction', foreign_keys=[bank_transaction_id])
    company_entry = db.relationship('CompanyFinancial', foreign_keys=[company_entry_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'bank_transaction_id': self.bank_transaction_id,
            'company_entry_id': self.company_entry_id,
            'match_score': self.match_score,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'bank_transaction': self.bank_transaction.to_dict() if self.bank_transaction else None,
            'company_entry': self.company_entry.to_dict() if self.company_entry else None
        }
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

