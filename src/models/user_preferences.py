from src.models.user import db
from datetime import datetime
from typing import Dict, Any, Optional
import json

class UserReconciliationConfig(db.Model):
    """
    User-specific reconciliation configuration preferences
    """
    __tablename__ = 'user_reconciliation_config'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    config_name = db.Column(db.String(100), nullable=False)  # e.g., 'default', 'strict', 'lenient'
    config_data = db.Column(db.Text, nullable=False)  # JSON string of configuration
    is_default = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='reconciliation_configs')
    
    def __repr__(self):
        return f'<UserReconciliationConfig {self.user_id}:{self.config_name}>'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'config_name': self.config_name,
            'config_data': self.get_config_data(),
            'is_default': self.is_default,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def get_config_data(self) -> Dict[str, Any]:
        """Parse JSON config data"""
        try:
            return json.loads(self.config_data)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_config_data(self, config_dict: Dict[str, Any]) -> None:
        """Set configuration data from dictionary"""
        self.config_data = json.dumps(config_dict)
    
    @classmethod
    def get_user_default_config(cls, user_id: int) -> Optional['UserReconciliationConfig']:
        """Get user's default configuration"""
        return cls.query.filter_by(
            user_id=user_id,
            is_default=True
        ).first()
    
    @classmethod
    def get_user_config_by_name(cls, user_id: int, config_name: str) -> Optional['UserReconciliationConfig']:
        """Get user's configuration by name"""
        return cls.query.filter_by(
            user_id=user_id,
            config_name=config_name
        ).first()
    
    @classmethod
    def get_all_user_configs(cls, user_id: int) -> list['UserReconciliationConfig']:
        """Get all configurations for a user"""
        return cls.query.filter_by(user_id=user_id).order_by(cls.config_name).all()

class UserReconciliationFeedback(db.Model):
    """
    Track user feedback on reconciliation decisions to improve matching
    """
    __tablename__ = 'user_reconciliation_feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    reconciliation_id = db.Column(db.Integer, db.ForeignKey('reconciliation_records.id'), nullable=True)
    feedback_type = db.Column(db.String(50), nullable=False)  # 'confirm', 'reject', 'adjust', 'manual_match'
    feedback_data = db.Column(db.Text, nullable=True)  # JSON string with detailed feedback
    original_score = db.Column(db.Float, nullable=True)  # Original match score
    adjusted_score = db.Column(db.Float, nullable=True)  # User-adjusted score
    confidence_rating = db.Column(db.Integer, nullable=True)  # 1-5 user confidence
    justification = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='reconciliation_feedback')
    reconciliation = db.relationship('ReconciliationRecord', backref='user_feedback')
    
    def __repr__(self):
        return f'<UserReconciliationFeedback {self.user_id}:{self.feedback_type}>'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'reconciliation_id': self.reconciliation_id,
            'feedback_type': self.feedback_type,
            'feedback_data': self.get_feedback_data(),
            'original_score': self.original_score,
            'adjusted_score': self.adjusted_score,
            'confidence_rating': self.confidence_rating,
            'justification': self.justification,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def get_feedback_data(self) -> Dict[str, Any]:
        """Parse JSON feedback data"""
        try:
            return json.loads(self.feedback_data) if self.feedback_data else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_feedback_data(self, feedback_dict: Dict[str, Any]) -> None:
        """Set feedback data from dictionary"""
        self.feedback_data = json.dumps(feedback_dict) if feedback_dict else None