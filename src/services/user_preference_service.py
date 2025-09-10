from typing import List, Dict, Any, Optional
from src.models.user_preferences import UserReconciliationConfig, UserReconciliationFeedback
from src.models.user import db
from src.services.reconciliation_service import ReconciliationConfig
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import (
    DatabaseError, ValidationError, RecordNotFoundError,
    ConfigurationError
)
from src.utils.error_handler import handle_service_errors, with_database_transaction
from datetime import datetime
import json

logger = get_logger(__name__)
audit_logger = get_audit_logger()

class UserPreferenceService:
    """
    Service for managing user reconciliation preferences and feedback
    """
    
    def __init__(self):
        pass
    
    @handle_service_errors('user_preference_service')
    @with_database_transaction
    def create_user_config(self, user_id: int, config_name: str, 
                          config_dict: Dict[str, Any], is_default: bool = False) -> UserReconciliationConfig:
        """
        Create a new user reconciliation configuration
        """
        if not user_id or user_id <= 0:
            raise ValidationError("Invalid user ID")
        
        if not config_name or not config_name.strip():
            raise ValidationError("Configuration name is required")
        
        if not config_dict:
            raise ValidationError("Configuration data is required")
        
        logger.info(f"Creating user config '{config_name}' for user {user_id}")
        
        try:
            # Check if config name already exists for this user
            existing_config = UserReconciliationConfig.get_user_config_by_name(user_id, config_name)
            if existing_config:
                raise ConfigurationError(f"Configuration '{config_name}' already exists for user {user_id}")
            
            # If this is set as default, remove default from other configs
            if is_default:
                existing_default = UserReconciliationConfig.get_user_default_config(user_id)
                if existing_default:
                    existing_default.is_default = False
            
            # Validate configuration data
            self._validate_config_data(config_dict)
            
            # Create new configuration
            user_config = UserReconciliationConfig(
                user_id=user_id,
                config_name=config_name,
                config_data=json.dumps(config_dict),
                is_default=is_default
            )
            
            db.session.add(user_config)
            
            logger.info(f"User config '{config_name}' created successfully for user {user_id}")
            audit_logger.log_user_operation('create_config', 1, 'success', {
                'user_id': user_id,
                'config_name': config_name,
                'is_default': is_default
            })
            
            return user_config
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError)):
                raise
            
            logger.error(f"Error creating user config: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to create user configuration: {str(e)}")
    
    @handle_service_errors('user_preference_service')
    @with_database_transaction
    def update_user_config(self, user_id: int, config_name: str, 
                         config_dict: Dict[str, Any], is_default: bool = None) -> UserReconciliationConfig:
        """
        Update an existing user reconciliation configuration
        """
        if not user_id or user_id <= 0:
            raise ValidationError("Invalid user ID")
        
        if not config_name or not config_name.strip():
            raise ValidationError("Configuration name is required")
        
        if not config_dict:
            raise ValidationError("Configuration data is required")
        
        logger.info(f"Updating user config '{config_name}' for user {user_id}")
        
        try:
            # Find existing configuration
            user_config = UserReconciliationConfig.get_user_config_by_name(user_id, config_name)
            if not user_config:
                raise RecordNotFoundError('UserReconciliationConfig', f"{user_id}:{config_name}")
            
            # Validate configuration data
            self._validate_config_data(config_dict)
            
            # Update configuration data
            user_config.set_config_data(config_dict)
            
            # Handle default status change
            if is_default is not None and is_default != user_config.is_default:
                if is_default:
                    # Remove default from other configs
                    existing_default = UserReconciliationConfig.get_user_default_config(user_id)
                    if existing_default and existing_default.id != user_config.id:
                        existing_default.is_default = False
                user_config.is_default = is_default
            
            # Update timestamp
            user_config.updated_at = datetime.utcnow()
            
            logger.info(f"User config '{config_name}' updated successfully for user {user_id}")
            audit_logger.log_user_operation('update_config', 1, 'success', {
                'user_id': user_id,
                'config_name': config_name,
                'is_default': user_config.is_default
            })
            
            return user_config
            
        except Exception as e:
            if isinstance(e, (ValidationError, ConfigurationError, RecordNotFoundError)):
                raise
            
            logger.error(f"Error updating user config: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to update user configuration: {str(e)}")
    
    @handle_service_errors('user_preference_service')
    def get_user_config(self, user_id: int, config_name: str = None) -> Optional[UserReconciliationConfig]:
        """
        Get user configuration by name, or default if no name specified
        """
        if not user_id or user_id <= 0:
            raise ValidationError("Invalid user ID")
        
        try:
            if config_name:
                user_config = UserReconciliationConfig.get_user_config_by_name(user_id, config_name)
            else:
                user_config = UserReconciliationConfig.get_user_default_config(user_id)
            
            return user_config
            
        except Exception as e:
            logger.error(f"Error getting user config: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get user configuration: {str(e)}")
    
    @handle_service_errors('user_preference_service')
    def get_all_user_configs(self, user_id: int) -> List[UserReconciliationConfig]:
        """
        Get all configurations for a user
        """
        if not user_id or user_id <= 0:
            raise ValidationError("Invalid user ID")
        
        try:
            configs = UserReconciliationConfig.get_all_user_configs(user_id)
            return configs
            
        except Exception as e:
            logger.error(f"Error getting all user configs: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get user configurations: {str(e)}")
    
    @handle_service_errors('user_preference_service')
    @with_database_transaction
    def delete_user_config(self, user_id: int, config_name: str) -> bool:
        """
        Delete a user configuration
        """
        if not user_id or user_id <= 0:
            raise ValidationError("Invalid user ID")
        
        if not config_name or not config_name.strip():
            raise ValidationError("Configuration name is required")
        
        logger.info(f"Deleting user config '{config_name}' for user {user_id}")
        
        try:
            user_config = UserReconciliationConfig.get_user_config_by_name(user_id, config_name)
            if not user_config:
                raise RecordNotFoundError('UserReconciliationConfig', f"{user_id}:{config_name}")
            
            db.session.delete(user_config)
            
            logger.info(f"User config '{config_name}' deleted successfully for user {user_id}")
            audit_logger.log_user_operation('delete_config', 1, 'success', {
                'user_id': user_id,
                'config_name': config_name
            })
            
            return True
            
        except Exception as e:
            if isinstance(e, (ValidationError, RecordNotFoundError)):
                raise
            
            logger.error(f"Error deleting user config: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to delete user configuration: {str(e)}")
    
    @handle_service_errors('user_preference_service')
    def get_reconciliation_config_for_user(self, user_id: int, config_name: str = None) -> ReconciliationConfig:
        """
        Get ReconciliationConfig object for user, with fallback to system defaults
        """
        try:
            # Try to get user config
            user_config = self.get_user_config(user_id, config_name)
            
            if user_config:
                # Create ReconciliationConfig from user preferences
                config_dict = user_config.get_config_data()
                config = ReconciliationConfig()
                config.update_from_dict(config_dict)
                logger.info(f"Using user config '{user_config.config_name}' for user {user_id}")
            else:
                # Use system defaults
                config = ReconciliationConfig()
                logger.info(f"Using default system config for user {user_id}")
            
            return config
            
        except Exception as e:
            logger.warning(f"Error getting user config, using defaults: {str(e)}")
            return ReconciliationConfig()
    
    @handle_service_errors('user_preference_service')
    @with_database_transaction
    def record_user_feedback(self, user_id: int, feedback_type: str, 
                           reconciliation_id: int = None, original_score: float = None,
                           adjusted_score: float = None, confidence_rating: int = None,
                           justification: str = None, feedback_data: Dict[str, Any] = None) -> UserReconciliationFeedback:
        """
        Record user feedback on reconciliation decisions
        """
        if not user_id or user_id <= 0:
            raise ValidationError("Invalid user ID")
        
        if not feedback_type or feedback_type not in ['confirm', 'reject', 'adjust', 'manual_match']:
            raise ValidationError("Invalid feedback type")
        
        if confidence_rating is not None and (confidence_rating < 1 or confidence_rating > 5):
            raise ValidationError("Confidence rating must be between 1 and 5")
        
        logger.info(f"Recording user feedback for user {user_id}: {feedback_type}")
        
        try:
            feedback = UserReconciliationFeedback(
                user_id=user_id,
                reconciliation_id=reconciliation_id,
                feedback_type=feedback_type,
                original_score=original_score,
                adjusted_score=adjusted_score,
                confidence_rating=confidence_rating,
                justification=justification
            )
            
            if feedback_data:
                feedback.set_feedback_data(feedback_data)
            
            db.session.add(feedback)
            
            logger.info(f"User feedback recorded successfully for user {user_id}")
            audit_logger.log_user_operation('record_feedback', 1, 'success', {
                'user_id': user_id,
                'feedback_type': feedback_type,
                'reconciliation_id': reconciliation_id,
                'confidence_rating': confidence_rating
            })
            
            return feedback
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            
            logger.error(f"Error recording user feedback: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to record user feedback: {str(e)}")
    
    @handle_service_errors('user_preference_service')
    def get_user_feedback_history(self, user_id: int, limit: int = 100) -> List[UserReconciliationFeedback]:
        """
        Get user's feedback history
        """
        if not user_id or user_id <= 0:
            raise ValidationError("Invalid user ID")
        
        try:
            feedback_history = UserReconciliationFeedback.query.filter_by(
                user_id=user_id
            ).order_by(
                UserReconciliationFeedback.created_at.desc()
            ).limit(limit).all()
            
            return feedback_history
            
        except Exception as e:
            logger.error(f"Error getting user feedback history: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get user feedback history: {str(e)}")
    
    @handle_service_errors('user_preference_service')
    def get_user_learning_data(self, user_id: int) -> Dict[str, Any]:
        """
        Get aggregated learning data from user feedback
        """
        if not user_id or user_id <= 0:
            raise ValidationError("Invalid user ID")
        
        try:
            feedback_history = self.get_user_feedback_history(user_id, limit=1000)
            
            learning_data = {
                'total_feedback_count': len(feedback_history),
                'feedback_types': {},
                'average_confidence': 0,
                'score_adjustments': [],
                'common_patterns': {}
            }
            
            confidence_ratings = []
            score_adjustments = []
            
            for feedback in feedback_history:
                # Count feedback types
                feedback_type = feedback.feedback_type
                learning_data['feedback_types'][feedback_type] = learning_data['feedback_types'].get(feedback_type, 0) + 1
                
                # Collect confidence ratings
                if feedback.confidence_rating:
                    confidence_ratings.append(feedback.confidence_rating)
                
                # Collect score adjustments
                if feedback.original_score is not None and feedback.adjusted_score is not None:
                    score_adjustments.append({
                        'original': feedback.original_score,
                        'adjusted': feedback.adjusted_score,
                        'difference': feedback.adjusted_score - feedback.original_score
                    })
            
            # Calculate average confidence
            if confidence_ratings:
                learning_data['average_confidence'] = sum(confidence_ratings) / len(confidence_ratings)
            
            # Calculate average adjustment
            if score_adjustments:
                avg_adjustment = sum(adj['difference'] for adj in score_adjustments) / len(score_adjustments)
                learning_data['average_score_adjustment'] = avg_adjustment
            
            learning_data['score_adjustments'] = score_adjustments[-20:]  # Last 20 adjustments
            
            return learning_data
            
        except Exception as e:
            logger.error(f"Error getting user learning data: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get user learning data: {str(e)}")
    
    def _validate_config_data(self, config_dict: Dict[str, Any]) -> None:
        """
        Validate configuration data
        """
        required_fields = [
            'amount_exact_weight', 'amount_close_weight', 'date_same_day_weight',
            'date_close_weight', 'description_weight', 'minimum_match_threshold'
        ]
        
        for field in required_fields:
            if field not in config_dict:
                raise ConfigurationError(f"Missing required field: {field}")
        
        # Validate numeric ranges
        if not (0 <= config_dict['minimum_match_threshold'] <= 1):
            raise ConfigurationError("minimum_match_threshold must be between 0 and 1")
        
        # Validate weights sum to approximately 1.0
        total_weight = (
            config_dict['amount_exact_weight'] + 
            config_dict['amount_close_weight'] + 
            config_dict['date_same_day_weight'] + 
            config_dict['date_close_weight'] + 
            config_dict['description_weight']
        )
        
        if abs(total_weight - 1.0) > 0.01:
            raise ConfigurationError(f"Weights must sum to 1.0, current sum: {total_weight}")