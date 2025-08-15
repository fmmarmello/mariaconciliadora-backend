from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from src.models.user import db

class CompanyFinancial(db.Model):
    __tablename__ = 'company_financial'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    description = db.Column(db.Text, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50))
    cost_center = db.Column(db.String(50))
    department = db.Column(db.String(50))
    project = db.Column(db.String(50))
    transaction_type = db.Column(db.String(10), nullable=False)  # 'expense' or 'income'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'description': self.description,
            'amount': self.amount,
            'category': self.category,
            'cost_center': self.cost_center,
            'department': self.department,
            'project': self.project,
            'transaction_type': self.transaction_type,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }