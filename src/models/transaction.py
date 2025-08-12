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
            'created_at': self.created_at.isoformat() if self.created_at else None
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
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'bank_name': self.bank_name,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'transactions_count': self.transactions_count,
            'status': self.status,
            'error_message': self.error_message
        }

