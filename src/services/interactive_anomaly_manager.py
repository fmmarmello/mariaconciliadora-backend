import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from src.models.transaction import ReconciliationRecord, Transaction
from src.models.company_financial import CompanyFinancial
from src.models.user import db
from src.models.user_preferences import UserReconciliationFeedback
from src.services.reconciliation_anomaly_detector import ReconciliationAnomalyDetector
from src.services.user_preference_service import UserPreferenceService
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import (
    ReconciliationError, ValidationError, RecordNotFoundError
)
from src.utils.error_handler import handle_service_errors, with_timeout

logger = get_logger(__name__)
audit_logger = get_audit_logger()

class InteractiveAnomalyManager:
    """
    Interactive anomaly management system with human supervision workflows
    """
    
    def __init__(self):
        self.detector = ReconciliationAnomalyDetector()
        self.user_preference_service = UserPreferenceService()
    
    @handle_service_errors('anomaly_management')
    def process_reconciliation_with_anomaly_detection(self, bank_transaction: Transaction, 
                                                    company_entry: CompanyFinancial,
                                                    match_score: float, 
                                                    score_breakdown: Dict[str, float],
                                                    user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a reconciliation match with comprehensive anomaly detection
        """
        # Perform anomaly analysis
        anomaly_analysis = self.detector.analyze_reconciliation_match(
            bank_transaction, company_entry, match_score, score_breakdown
        )
        
        # Create reconciliation record with anomaly data
        reconciliation_record = ReconciliationRecord(
            bank_transaction_id=bank_transaction.id,
            company_entry_id=company_entry.id,
            match_score=match_score,
            status='pending',
            score_breakdown=json.dumps(score_breakdown),
            confidence_level=anomaly_analysis['confidence_level'],
            risk_factors=json.dumps(anomaly_analysis['risk_factors'])
        )
        
        # Add anomaly information if detected
        if anomaly_analysis['is_anomaly']:
            reconciliation_record.is_anomaly = True
            reconciliation_record.anomaly_type = self._get_primary_anomaly_type(anomaly_analysis['anomalies'])
            reconciliation_record.anomaly_severity = anomaly_analysis['overall_severity']
            reconciliation_record.anomaly_score = anomaly_analysis['total_anomaly_score']
            reconciliation_record.anomaly_reason = self._generate_anomaly_summary(anomaly_analysis)
            reconciliation_record.anomaly_detected_at = datetime.utcnow()
            
            # Set initial action based on severity
            if anomaly_analysis['human_review_required']:
                reconciliation_record.anomaly_action = 'pending_review'
            else:
                reconciliation_record.anomaly_action = 'auto_flagged'
        
        db.session.add(reconciliation_record)
        db.session.commit()
        
        # Log the anomaly detection
        if anomaly_analysis['is_anomaly']:
            audit_logger.log_anomaly_detection(
                reconciliation_record.id,
                anomaly_analysis['overall_severity'],
                len(anomaly_analysis['anomalies']),
                user_id
            )
            
            # If human review is required, create a notification
            if anomaly_analysis['human_review_required'] and user_id:
                self._create_anomaly_notification(reconciliation_record, user_id, anomaly_analysis)
        
        return {
            'success': True,
            'reconciliation_id': reconciliation_record.id,
            'anomaly_detected': anomaly_analysis['is_anomaly'],
            'anomaly_analysis': anomaly_analysis,
            'human_action_required': anomaly_analysis['human_review_required'],
            'priority': anomaly_analysis['priority_score']
        }
    
    @handle_service_errors('anomaly_review')
    def review_anomaly(self, reconciliation_id: int, user_id: int, 
                       action: str, justification: str, 
                       additional_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle human review of an anomaly with comprehensive workflow
        """
        # Validate action
        valid_actions = ['confirm', 'dismiss', 'escalate', 'request_info', 'mark_for_training']
        if action not in valid_actions:
            raise ValidationError(f"Invalid action. Must be one of: {valid_actions}")
        
        # Get the reconciliation record
        record = ReconciliationRecord.query.get(reconciliation_id)
        if not record:
            raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
        
        if not record.is_anomaly:
            raise ValidationError("Record is not flagged as an anomaly")
        
        # Record user action
        old_action = record.anomaly_action or 'pending'
        record.resolve_anomaly(action, justification, user_id)
        
        # Handle specific actions
        result = {'success': True, 'action_taken': action}
        
        if action == 'confirm':
            # User confirms this is indeed an anomaly
            record.status = 'confirmed'
            result['message'] = 'Anomaly confirmed and reconciliation flagged'
            
            # Create feedback record for learning
            self._record_user_feedback(user_id, reconciliation_id, 'confirm', {
                'anomaly_confirmed': True,
                'justification': justification,
                'additional_data': additional_data
            })
        
        elif action == 'dismiss':
            # User dismisses the anomaly flag
            record.is_anomaly = False
            record.status = 'confirmed'  # Auto-confirm dismissed anomalies
            result['message'] = 'Anomaly dismissed and reconciliation confirmed'
            
            # Create feedback record for learning
            self._record_user_feedback(user_id, reconciliation_id, 'dismiss', {
                'anomaly_dismissed': True,
                'justification': justification,
                'additional_data': additional_data
            })
        
        elif action == 'escalate':
            # Escalate to higher authority
            record.anomaly_severity = 'critical'
            result['message'] = 'Anomaly escalated for review'
            result['escalated'] = True
            
            # Create escalation record
            self._create_escalation_record(record, user_id, justification, additional_data)
        
        elif action == 'request_info':
            # Request additional information
            result['message'] = 'Additional information requested'
            result['info_requested'] = True
            
            # Create info request
            self._create_info_request(record, user_id, justification, additional_data)
        
        elif action == 'mark_for_training':
            # Mark for model training/improvement
            result['message'] = 'Anomaly marked for model training'
            result['training_flagged'] = True
            
            # Create training record
            self._create_training_record(record, user_id, justification, additional_data)
        
        db.session.commit()
        
        # Log the review action
        audit_logger.log_anomaly_review(
            reconciliation_id, user_id, old_action, action, justification
        )
        
        # Check if this affects other similar anomalies
        if action in ['confirm', 'dismiss']:
            self._update_similar_anomalies(record, user_id, action, justification)
        
        return result
    
    @handle_service_errors('anomaly_workflow')
    def get_anomaly_workflow_suggestions(self, reconciliation_id: int, user_id: int) -> Dict[str, Any]:
        """
        Get AI-powered suggestions for handling an anomaly based on historical patterns
        """
        record = ReconciliationRecord.query.get(reconciliation_id)
        if not record:
            raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
        
        if not record.is_anomaly:
            raise ValidationError("Record is not flagged as an anomaly")
        
        # Get user's historical handling patterns
        user_history = self._get_user_anomaly_history(user_id, record.anomaly_type)
        
        # Get similar anomalies and their resolutions
        similar_anomalies = self._find_similar_anomalies(record)
        
        # Generate suggestions based on patterns
        suggestions = self._generate_workflow_suggestions(record, user_history, similar_anomalies)
        
        # Get recommended actions with confidence scores
        recommended_actions = self._get_recommended_actions(record, user_history, similar_anomalies)
        
        return {
            'reconciliation_id': reconciliation_id,
            'anomaly_type': record.anomaly_type,
            'severity': record.anomaly_severity,
            'user_history_stats': user_history,
            'similar_anomalies_count': len(similar_anomalies),
            'similar_anomalies_resolution': self._summarize_similar_resolutions(similar_anomalies),
            'suggestions': suggestions,
            'recommended_actions': recommended_actions,
            'estimated_resolution_time': self._estimate_resolution_time(record, user_history),
            'complexity_level': self._assess_complexity(record, similar_anomalies)
        }
    
    @handle_service_errors('anomaly_batch')
    def batch_process_anomalies(self, anomaly_ids: List[int], user_id: int, 
                              batch_action: str, batch_justification: str,
                              filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process multiple anomalies in batch with human supervision
        """
        if not anomaly_ids:
            raise ValidationError("No anomaly IDs provided")
        
        valid_batch_actions = ['confirm_all', 'dismiss_all', 'escalate_high_severity', 'request_reviews']
        if batch_action not in valid_batch_actions:
            raise ValidationError(f"Invalid batch action: {batch_action}")
        
        # Get the anomaly records
        query = ReconciliationRecord.query.filter(
            ReconciliationRecord.id.in_(anomaly_ids),
            ReconciliationRecord.is_anomaly == True
        )
        
        # Apply additional filters if provided
        if filters:
            if 'severity' in filters:
                query = query.filter(ReconciliationRecord.anomaly_severity.in_(filters['severity']))
            if 'type' in filters:
                query = query.filter(ReconciliationRecord.anomaly_type.in_(filters['type']))
            if 'date_range' in filters:
                query = query.filter(ReconciliationRecord.anomaly_detected_at >= filters['date_range']['start'])
                query = query.filter(ReconciliationRecord.anomaly_detected_at <= filters['date_range']['end'])
        
        records = query.all()
        
        results = {
            'total_processed': len(records),
            'success_count': 0,
            'error_count': 0,
            'errors': [],
            'details': []
        }
        
        for record in records:
            try:
                # Apply batch action logic
                if batch_action == 'confirm_all':
                    action = 'confirm'
                elif batch_action == 'dismiss_all':
                    action = 'dismiss'
                elif batch_action == 'escalate_high_severity':
                    action = 'escalate' if record.anomaly_severity in ['high', 'critical'] else 'confirm'
                elif batch_action == 'request_reviews':
                    action = 'request_info'
                
                # Process individual anomaly
                result = self.review_anomaly(
                    record.id, user_id, action, batch_justification
                )
                
                results['success_count'] += 1
                results['details'].append({
                    'id': record.id,
                    'action': action,
                    'success': True
                })
                
            except Exception as e:
                results['error_count'] += 1
                results['errors'].append({
                    'id': record.id,
                    'error': str(e)
                })
                results['details'].append({
                    'id': record.id,
                    'action': 'error',
                    'success': False,
                    'error': str(e)
                })
        
        # Log batch processing
        audit_logger.log_batch_anomaly_processing(
            user_id, batch_action, results['success_count'], results['error_count']
        )
        
        return results
    
    @handle_service_errors('anomaly_insights')
    def get_anomaly_insights_dashboard(self, user_id: Optional[int] = None, 
                                     time_range: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive anomaly insights for dashboard
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_range)
        
        # Get base statistics
        stats = self.detector.get_anomaly_statistics(user_id)
        
        # Get time-based trends
        trends = self._get_anomaly_trends(start_date, end_date, user_id)
        
        # Get user performance metrics
        user_metrics = self._get_user_performance_metrics(user_id, start_date, end_date) if user_id else {}
        
        # Get anomaly hotspots
        hotspots = self._get_anomaly_hotspots(start_date, end_date, user_id)
        
        # Get recommendations for system improvement
        recommendations = self._generate_system_recommendations(stats, trends, user_metrics)
        
        return {
            'summary': stats,
            'trends': trends,
            'user_metrics': user_metrics,
            'hotspots': hotspots,
            'recommendations': recommendations,
            'time_range_days': time_range,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _get_primary_anomaly_type(self, anomalies: List[Dict]) -> str:
        """Get the primary anomaly type from multiple anomalies"""
        if not anomalies:
            return 'unknown'
        
        # Sort by severity and score to get primary anomaly
        severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        
        sorted_anomalies = sorted(anomalies, key=lambda x: (
            severity_weights.get(x['severity'], 0),
            x['score']
        ), reverse=True)
        
        return sorted_anomalies[0]['type']
    
    def _generate_anomaly_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable anomaly summary"""
        if not analysis['anomalies']:
            return "No anomalies detected"
        
        summary_parts = []
        
        # Add severity info
        summary_parts.append(f"Severity: {analysis['overall_severity'].upper()}")
        
        # Add count of anomalies
        count = len(analysis['anomalies'])
        summary_parts.append(f"{count} anomaly type(s) detected")
        
        # Add top anomaly types
        anomaly_types = [a['type'] for a in analysis['anomalies'][:2]]
        summary_parts.append(f"Main issues: {', '.join(anomaly_types)}")
        
        return " | ".join(summary_parts)
    
    def _create_anomaly_notification(self, record: ReconciliationRecord, user_id: int, 
                                   analysis: Dict[str, Any]):
        """Create notification for human review"""
        # This would integrate with your notification system
        logger.info(f"Anomaly notification created for record {record.id}, user {user_id}")
    
    def _record_user_feedback(self, user_id: int, reconciliation_id: int, 
                             feedback_type: str, feedback_data: Dict):
        """Record user feedback for learning"""
        try:
            self.user_preference_service.record_user_feedback(
                user_id=user_id,
                feedback_type=feedback_type,
                reconciliation_id=reconciliation_id,
                confidence_rating=feedback_data.get('confidence_rating', 5),
                justification=feedback_data.get('justification', ''),
                feedback_data=feedback_data
            )
        except Exception as e:
            logger.warning(f"Failed to record user feedback: {str(e)}")
    
    def _create_escalation_record(self, record: ReconciliationRecord, user_id: int, 
                                 justification: str, additional_data: Optional[Dict]):
        """Create escalation record"""
        logger.info(f"Anomaly {record.id} escalated by user {user_id}")
    
    def _create_info_request(self, record: ReconciliationRecord, user_id: int, 
                           justification: str, additional_data: Optional[Dict]):
        """Create information request"""
        logger.info(f"Information requested for anomaly {record.id} by user {user_id}")
    
    def _create_training_record(self, record: ReconciliationRecord, user_id: int, 
                              justification: str, additional_data: Optional[Dict]):
        """Create training record for model improvement"""
        logger.info(f"Anomaly {record.id} marked for training by user {user_id}")
    
    def _update_similar_anomalies(self, record: ReconciliationRecord, user_id: int, 
                                 action: str, justification: str):
        """Update similar anomalies based on user action"""
        # Find similar anomalies and apply learning
        similar = self._find_similar_anomalies(record, limit=10)
        
        for similar_record in similar:
            if similar_record.anomaly_action in ['pending', None]:
                # Apply similar action with reduced confidence
                confidence_factor = 0.7  # 70% confidence in applying similar action
                
                if action == 'confirm' and similar_record.anomaly_score > 0.8:
                    similar_record.resolve_anomaly(action, 
                        f"Auto-resolved based on similar anomaly (ID: {record.id}): {justification}", 
                        user_id)
                elif action == 'dismiss' and similar_record.anomaly_score < 0.4:
                    similar_record.resolve_anomaly(action,
                        f"Auto-dismissed based on similar anomaly (ID: {record.id}): {justification}",
                        user_id)
        
        db.session.commit()
    
    def _get_user_anomaly_history(self, user_id: int, anomaly_type: str) -> Dict[str, Any]:
        """Get user's historical handling of specific anomaly types"""
        try:
            feedback = UserReconciliationFeedback.query.filter(
                UserReconciliationFeedback.user_id == user_id,
                UserReconciliationFeedback.feedback_type.in_(['confirm', 'dismiss'])
            ).all()
            
            type_specific = [f for f in feedback if anomaly_type in f.feedback_data.get('anomaly_types', [])]
            
            return {
                'total_handled': len(feedback),
                'type_specific_handled': len(type_specific),
                'confirmation_rate': len([f for f in feedback if f.feedback_type == 'confirm']) / len(feedback) if feedback else 0,
                'avg_confidence_rating': sum(f.confidence_rating or 0 for f in feedback) / len(feedback) if feedback else 0
            }
        except Exception as e:
            logger.warning(f"Error getting user history: {str(e)}")
            return {'total_handled': 0, 'type_specific_handled': 0, 'confirmation_rate': 0, 'avg_confidence_rating': 0}
    
    def _find_similar_anomalies(self, record: ReconciliationRecord, limit: int = 5) -> List[ReconciliationRecord]:
        """Find similar anomalies based on type and characteristics"""
        similar = ReconciliationRecord.query.filter(
            ReconciliationRecord.id != record.id,
            ReconciliationRecord.is_anomaly == True,
            ReconciliationRecord.anomaly_type == record.anomaly_type,
            ReconciliationRecord.anomaly_detected_at >= datetime.utcnow() - timedelta(days=90)
        ).order_by(ReconciliationRecord.anomaly_detected_at.desc()).limit(limit).all()
        
        return similar
    
    def _generate_workflow_suggestions(self, record: ReconciliationRecord, 
                                    user_history: Dict, similar_anomalies: List) -> List[str]:
        """Generate workflow suggestions based on patterns"""
        suggestions = []
        
        # Based on user history
        if user_history['total_handled'] > 5:
            if user_history['confirmation_rate'] > 0.8:
                suggestions.append("Based on your history, you tend to confirm this type of anomaly")
            elif user_history['confirmation_rate'] < 0.2:
                suggestions.append("Based on your history, you tend to dismiss this type of anomaly")
        
        # Based on similar anomalies
        if similar_anomalies:
            confirmed_count = len([a for a in similar_anomalies if a.anomaly_action == 'confirm'])
            if confirmed_count > len(similar_anomalies) * 0.7:
                suggestions.append("Similar anomalies were usually confirmed")
            elif confirmed_count < len(similar_anomalies) * 0.3:
                suggestions.append("Similar anomalies were usually dismissed")
        
        # Based on severity
        if record.anomaly_severity in ['critical', 'high']:
            suggestions.append("Consider escalating this anomaly due to high severity")
        elif record.anomaly_severity == 'low':
            suggestions.append("This low-severity anomaly may be safe to dismiss")
        
        return suggestions
    
    def _get_recommended_actions(self, record: ReconciliationRecord, 
                               user_history: Dict, similar_anomalies: List) -> List[Dict]:
        """Get recommended actions with confidence scores"""
        actions = []
        
        # Base recommendations on severity
        if record.anomaly_severity == 'critical':
            actions.append({'action': 'escalate', 'confidence': 0.9, 'reason': 'Critical severity requires escalation'})
            actions.append({'action': 'confirm', 'confidence': 0.8, 'reason': 'Critical anomaly should be confirmed'})
        elif record.anomaly_severity == 'high':
            actions.append({'action': 'confirm', 'confidence': 0.7, 'reason': 'High severity anomaly likely real'})
            actions.append({'action': 'request_info', 'confidence': 0.6, 'reason': 'May need additional information'})
        elif record.anomaly_severity == 'low':
            actions.append({'action': 'dismiss', 'confidence': 0.6, 'reason': 'Low severity anomaly may be false positive'})
            actions.append({'action': 'confirm', 'confidence': 0.4, 'reason': 'Still worth confirming'})
        
        # Adjust based on user patterns
        if user_history['total_handled'] > 3:
            if user_history['confirmation_rate'] > 0.7:
                # User tends to confirm - boost confirm confidence
                for action in actions:
                    if action['action'] == 'confirm':
                        action['confidence'] = min(1.0, action['confidence'] + 0.2)
            elif user_history['confirmation_rate'] < 0.3:
                # User tends to dismiss - boost dismiss confidence
                for action in actions:
                    if action['action'] == 'dismiss':
                        action['confidence'] = min(1.0, action['confidence'] + 0.2)
        
        # Sort by confidence
        actions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return actions
    
    def _estimate_resolution_time(self, record: ReconciliationRecord, user_history: Dict) -> str:
        """Estimate time required for resolution"""
        base_times = {
            'low': '5-10 minutes',
            'medium': '15-30 minutes', 
            'high': '30-60 minutes',
            'critical': '1-2 hours'
        }
        
        base_time = base_times.get(record.anomaly_severity, '30 minutes')
        
        # Adjust based on user experience
        if user_history['total_handled'] > 10:
            return f"~{int(int(base_time.split('-')[1].split()[0]) * 0.7)} minutes (experienced user)"
        elif user_history['total_handled'] < 3:
            return f"~{int(int(base_time.split('-')[1].split()[0]) * 1.5)} minutes (new user)"
        
        return base_time
    
    def _assess_complexity(self, record: ReconciliationRecord, similar_anomalies: List) -> str:
        """Assess complexity of anomaly resolution"""
        complexity_score = 0
        
        # Base complexity on severity
        severity_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        complexity_score += severity_scores.get(record.anomaly_severity, 2)
        
        # Adjust based on anomaly type
        complex_types = ['amount_mismatch', 'compliance_risk', 'unusual_pattern']
        if record.anomaly_type in complex_types:
            complexity_score += 1
        
        # Adjust based on similar anomalies resolution
        if similar_anomalies:
            varied_actions = len(set(a.anomaly_action for a in similar_anomalies if a.anomaly_action))
            if varied_actions > 2:
                complexity_score += 1
        
        if complexity_score <= 2:
            return 'simple'
        elif complexity_score <= 4:
            return 'moderate'
        else:
            return 'complex'
    
    def _get_anomaly_trends(self, start_date: datetime, end_date: datetime, 
                           user_id: Optional[int]) -> Dict[str, Any]:
        """Get anomaly trends over time"""
        query = ReconciliationRecord.query.filter(
            ReconciliationRecord.anomaly_detected_at >= start_date,
            ReconciliationRecord.anomaly_detected_at <= end_date
        )
        
        if user_id:
            query = query.filter(ReconciliationRecord.anomaly_reviewed_by == user_id)
        
        records = query.all()
        
        # Group by day
        daily_counts = {}
        for record in records:
            day = record.anomaly_detected_at.date().isoformat()
            daily_counts[day] = daily_counts.get(day, 0) + 1
        
        return {
            'daily_counts': daily_counts,
            'total_in_period': len(records),
            'trend_direction': self._calculate_trend_direction(list(daily_counts.values()))
        }
    
    def _get_user_performance_metrics(self, user_id: int, start_date: datetime, 
                                    end_date: datetime) -> Dict[str, Any]:
        """Get user performance metrics for anomaly handling"""
        records = ReconciliationRecord.query.filter(
            ReconciliationRecord.anomaly_reviewed_by == user_id,
            ReconciliationRecord.anomaly_reviewed_at >= start_date,
            ReconciliationRecord.anomaly_reviewed_at <= end_date
        ).all()
        
        if not records:
            return {}
        
        resolution_times = []
        confirmed_count = 0
        dismissed_count = 0
        
        for record in records:
            if record.anomaly_reviewed_at and record.anomaly_detected_at:
                resolution_time = (record.anomaly_reviewed_at - record.anomaly_detected_at).total_seconds() / 3600
                resolution_times.append(resolution_time)
            
            if record.anomaly_action == 'confirm':
                confirmed_count += 1
            elif record.anomaly_action == 'dismiss':
                dismissed_count += 1
        
        return {
            'total_handled': len(records),
            'avg_resolution_time_hours': sum(resolution_times) / len(resolution_times) if resolution_times else 0,
            'confirmation_rate': confirmed_count / len(records) if records else 0,
            'dismissal_rate': dismissed_count / len(records) if records else 0
        }
    
    def _get_anomaly_hotspots(self, start_date: datetime, end_date: datetime, 
                            user_id: Optional[int]) -> List[Dict]:
        """Get anomaly hotspots (areas with high anomaly concentration)"""
        # This would analyze patterns like specific transaction types, amounts, dates, etc.
        # For now, return basic statistics
        return []
    
    def _generate_system_recommendations(self, stats: Dict, trends: Dict, 
                                        user_metrics: Dict) -> List[str]:
        """Generate system improvement recommendations"""
        recommendations = []
        
        # Based on anomaly volume
        if stats['total_anomalies'] > 100:
            recommendations.append("Consider implementing automated rules for common anomaly types")
        
        # Based on resolution times
        if user_metrics.get('avg_resolution_time_hours', 0) > 24:
            recommendations.append("Anomaly resolution times are high - consider workflow improvements")
        
        # Based on confirmation rates
        if user_metrics.get('confirmation_rate', 0) > 0.8:
            recommendations.append("High confirmation rate may indicate detection is too sensitive")
        elif user_metrics.get('confirmation_rate', 0) < 0.2:
            recommendations.append("Low confirmation rate may indicate detection is missing real issues")
        
        return recommendations
    
    def _calculate_trend_direction(self, values: List[int]) -> str:
        """Calculate trend direction from time series values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple trend calculation
        recent_avg = sum(values[-3:]) / min(3, len(values))
        older_avg = sum(values[:-3]) / max(1, len(values) - 3)
        
        if recent_avg > older_avg * 1.2:
            return 'increasing'
        elif recent_avg < older_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _summarize_similar_resolutions(self, similar_anomalies: List[ReconciliationRecord]) -> Dict[str, int]:
        """Summarize how similar anomalies were resolved"""
        summary = {'confirm': 0, 'dismiss': 0, 'escalate': 0, 'pending': 0}
        
        for record in similar_anomalies:
            if record.anomaly_action:
                summary[record.anomaly_action] = summary.get(record.anomaly_action, 0) + 1
            else:
                summary['pending'] += 1
        
        return summary