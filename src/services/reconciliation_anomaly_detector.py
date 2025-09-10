import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from src.models.transaction import ReconciliationRecord, Transaction
from src.models.company_financial import CompanyFinancial
from src.models.user import db
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import (
    ReconciliationError, ValidationError, InsufficientDataError
)
from src.utils.error_handler import handle_service_errors, with_timeout

logger = get_logger(__name__)
audit_logger = get_audit_logger()

class ReconciliationAnomalyDetector:
    """
    Advanced anomaly detection system for reconciliation records with human supervision
    """
    
    def __init__(self):
        self.anomaly_types = {
            'amount_mismatch': 'Significant difference between transaction amounts',
            'date_discrepancy': 'Unusual date difference between records',
            'description_mismatch': 'Poor description similarity between records',
            'timing_anomaly': 'Unusual timing patterns in reconciliation',
            'category_conflict': 'Conflicting categories between bank and company records',
            'duplicate_match': 'Potential duplicate matching',
            'high_risk_transaction': 'Transaction flagged as high risk',
            'unusual_pattern': 'Unusual pattern in reconciliation history',
            'missing_data': 'Critical data missing in reconciliation',
            'compliance_risk': 'Potential compliance or regulatory issues'
        }
        
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.95
        }
    
    @handle_service_errors('anomaly_detection')
    def analyze_reconciliation_match(self, bank_transaction: Transaction, company_entry: CompanyFinancial, 
                                    match_score: float, score_breakdown: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze a reconciliation match for potential anomalies with detailed scoring
        """
        anomalies = []
        risk_factors = []
        total_anomaly_score = 0.0
        
        # 1. Amount Anomaly Detection
        amount_analysis = self._analyze_amount_anomaly(bank_transaction, company_entry)
        if amount_analysis['is_anomaly']:
            anomalies.append({
                'type': 'amount_mismatch',
                'severity': amount_analysis['severity'],
                'score': amount_analysis['score'],
                'reason': amount_analysis['reason'],
                'details': amount_analysis['details']
            })
            total_anomaly_score += amount_analysis['score'] * self._get_severity_weight(amount_analysis['severity'])
            risk_factors.append('amount_significant_discrepancy')
        
        # 2. Date Anomaly Detection
        date_analysis = self._analyze_date_anomaly(bank_transaction, company_entry)
        if date_analysis['is_anomaly']:
            anomalies.append({
                'type': 'date_discrepancy',
                'severity': date_analysis['severity'],
                'score': date_analysis['score'],
                'reason': date_analysis['reason'],
                'details': date_analysis['details']
            })
            total_anomaly_score += date_analysis['score'] * self._get_severity_weight(date_analysis['severity'])
            risk_factors.append('date_unusual_difference')
        
        # 3. Description Similarity Analysis
        description_analysis = self._analyze_description_similarity(bank_transaction, company_entry)
        if description_analysis['is_anomaly']:
            anomalies.append({
                'type': 'description_mismatch',
                'severity': description_analysis['severity'],
                'score': description_analysis['score'],
                'reason': description_analysis['reason'],
                'details': description_analysis['details']
            })
            total_anomaly_score += description_analysis['score'] * self._get_severity_weight(description_analysis['severity'])
            risk_factors.append('description_poor_match')
        
        # 4. Category Conflict Detection
        category_analysis = self._analyze_category_conflict(bank_transaction, company_entry)
        if category_analysis['is_anomaly']:
            anomalies.append({
                'type': 'category_conflict',
                'severity': category_analysis['severity'],
                'score': category_analysis['score'],
                'reason': category_analysis['reason'],
                'details': category_analysis['details']
            })
            total_anomaly_score += category_analysis['score'] * self._get_severity_weight(category_analysis['severity'])
            risk_factors.append('category_mismatch')
        
        # 5. Risk Pattern Analysis
        pattern_analysis = self._analyze_risk_patterns(bank_transaction, company_entry)
        if pattern_analysis['is_anomaly']:
            anomalies.append({
                'type': 'unusual_pattern',
                'severity': pattern_analysis['severity'],
                'score': pattern_analysis['score'],
                'reason': pattern_analysis['reason'],
                'details': pattern_analysis['details']
            })
            total_anomaly_score += pattern_analysis['score'] * self._get_severity_weight(pattern_analysis['severity'])
            risk_factors.extend(pattern_analysis['risk_factors'])
        
        # 6. Duplicate Detection
        duplicate_analysis = self._detect_duplicate_match(bank_transaction, company_entry)
        if duplicate_analysis['is_anomaly']:
            anomalies.append({
                'type': 'duplicate_match',
                'severity': duplicate_analysis['severity'],
                'score': duplicate_analysis['score'],
                'reason': duplicate_analysis['reason'],
                'details': duplicate_analysis['details']
            })
            total_anomaly_score += duplicate_analysis['score'] * self._get_severity_weight(duplicate_analysis['severity'])
            risk_factors.append('potential_duplicate')
        
        # Calculate overall anomaly assessment
        overall_severity = self._calculate_overall_severity(total_anomaly_score, len(anomalies))
        confidence_level = self._calculate_confidence_level(match_score, total_anomaly_score, anomalies)
        
        return {
            'is_anomaly': len(anomalies) > 0,
            'anomaly_count': len(anomalies),
            'total_anomaly_score': min(total_anomaly_score, 1.0),
            'overall_severity': overall_severity,
            'confidence_level': confidence_level,
            'anomalies': anomalies,
            'risk_factors': list(set(risk_factors)),
            'recommendations': self._generate_recommendations(anomalies, overall_severity),
            'human_review_required': self._requires_human_review(anomalies, overall_severity),
            'priority_score': self._calculate_priority_score(total_anomaly_score, overall_severity, anomalies)
        }
    
    def _analyze_amount_anomaly(self, bank_tx: Transaction, company_entry: CompanyFinancial) -> Dict[str, Any]:
        """Analyze amount discrepancies between bank and company records"""
        bank_amount = abs(bank_tx.amount)
        company_amount = abs(company_entry.amount)
        
        if bank_amount == 0 or company_amount == 0:
            return {
                'is_anomaly': True,
                'severity': 'critical',
                'score': 0.9,
                'reason': 'Zero amount detected in one or both records',
                'details': {
                    'bank_amount': bank_amount,
                    'company_amount': company_amount
                }
            }
        
        amount_diff = abs(bank_amount - company_amount)
        amount_diff_percent = amount_diff / max(bank_amount, company_amount)
        
        # Severity thresholds for amount differences
        if amount_diff_percent > 0.5:  # > 50% difference
            severity = 'critical'
            score = min(0.95, amount_diff_percent * 1.5)
        elif amount_diff_percent > 0.2:  # > 20% difference
            severity = 'high'
            score = min(0.8, amount_diff_percent * 2)
        elif amount_diff_percent > 0.05:  # > 5% difference
            severity = 'medium'
            score = min(0.6, amount_diff_percent * 5)
        elif amount_diff_percent > 0.01:  # > 1% difference
            severity = 'low'
            score = min(0.3, amount_diff_percent * 10)
        else:
            return {'is_anomaly': False}
        
        return {
            'is_anomaly': True,
            'severity': severity,
            'score': score,
            'reason': f'Significant amount difference: {amount_diff_percent:.1%}',
            'details': {
                'bank_amount': bank_amount,
                'company_amount': company_amount,
                'difference': amount_diff,
                'difference_percent': amount_diff_percent
            }
        }
    
    def _analyze_date_anomaly(self, bank_tx: Transaction, company_entry: CompanyFinancial) -> Dict[str, Any]:
        """Analyze date discrepancies between records"""
        if not bank_tx.date or not company_entry.date:
            return {
                'is_anomaly': True,
                'severity': 'high',
                'score': 0.8,
                'reason': 'Missing date in one or both records',
                'details': {
                    'bank_date': bank_tx.date.isoformat() if bank_tx.date else None,
                    'company_date': company_entry.date.isoformat() if company_entry.date else None
                }
            }
        
        date_diff = abs((bank_tx.date - company_entry.date).days)
        
        # Severity thresholds for date differences
        if date_diff > 90:  # > 3 months
            severity = 'critical'
            score = min(0.95, date_diff / 365)
        elif date_diff > 30:  # > 1 month
            severity = 'high'
            score = min(0.8, date_diff / 120)
        elif date_diff > 7:  # > 1 week
            severity = 'medium'
            score = min(0.6, date_diff / 60)
        elif date_diff > 3:  # > 3 days
            severity = 'low'
            score = min(0.3, date_diff / 30)
        else:
            return {'is_anomaly': False}
        
        return {
            'is_anomaly': True,
            'severity': severity,
            'score': score,
            'reason': f'Unusual date difference: {date_diff} days',
            'details': {
                'bank_date': bank_tx.date.isoformat(),
                'company_date': company_entry.date.isoformat(),
                'date_difference_days': date_diff
            }
        }
    
    def _analyze_description_similarity(self, bank_tx: Transaction, company_entry: CompanyFinancial) -> Dict[str, Any]:
        """Analyze description similarity using advanced text matching"""
        from difflib import SequenceMatcher
        
        bank_desc = bank_tx.description.lower().strip()
        company_desc = company_entry.description.lower().strip()
        
        if not bank_desc or not company_desc:
            return {
                'is_anomaly': True,
                'severity': 'medium',
                'score': 0.6,
                'reason': 'Missing description in one or both records',
                'details': {
                    'bank_description': bank_tx.description,
                    'company_description': company_entry.description
                }
            }
        
        # Calculate similarity score
        similarity = SequenceMatcher(None, bank_desc, company_desc).ratio()
        
        # Extract key terms for comparison
        bank_terms = set(bank_desc.split())
        company_terms = set(company_desc.split())
        common_terms = bank_terms.intersection(company_terms)
        
        # Severity based on similarity and common terms
        if similarity < 0.1 and len(common_terms) == 0:
            severity = 'high'
            score = 0.8
        elif similarity < 0.3 and len(common_terms) <= 1:
            severity = 'medium'
            score = 0.6
        elif similarity < 0.5:
            severity = 'low'
            score = 0.3
        else:
            return {'is_anomaly': False}
        
        return {
            'is_anomaly': True,
            'severity': severity,
            'score': score,
            'reason': f'Poor description similarity: {similarity:.1%}',
            'details': {
                'similarity_score': similarity,
                'common_terms': list(common_terms),
                'bank_terms_count': len(bank_terms),
                'company_terms_count': len(company_terms)
            }
        }
    
    def _analyze_category_conflict(self, bank_tx: Transaction, company_entry: CompanyFinancial) -> Dict[str, Any]:
        """Analyze category conflicts between bank and company records"""
        bank_category = bank_tx.category.lower() if bank_tx.category else 'outros'
        company_category = company_entry.category.lower() if company_entry.category else 'outros'
        
        # Define category conflict matrix
        category_conflicts = {
            ('alimentacao', 'transporte'): True,
            ('alimentacao', 'vestuario'): True,
            ('transporte', 'educacao'): True,
            ('saude', 'lazer'): True,
            ('investimento', 'saude'): True,
            ('salario', 'transferencia'): True,
        }
        
        # Check for direct conflict
        has_conflict = category_conflicts.get((bank_category, company_category), False) or \
                      category_conflicts.get((company_category, bank_category), False)
        
        if has_conflict:
            return {
                'is_anomaly': True,
                'severity': 'medium',
                'score': 0.5,
                'reason': f'Category conflict between bank and company records',
                'details': {
                    'bank_category': bank_category,
                    'company_category': company_category,
                    'conflict_type': 'cross_category_mismatch'
                }
            }
        
        # Check for generic vs specific category mismatch
        if bank_category != company_category:
            generic_categories = ['outros', 'transferencia', 'saque']
            if bank_category in generic_categories or company_category in generic_categories:
                return {
                    'is_anomaly': True,
                    'severity': 'low',
                    'score': 0.3,
                    'reason': f'Category mismatch with generic category',
                    'details': {
                        'bank_category': bank_category,
                        'company_category': company_category,
                        'conflict_type': 'generic_specific_mismatch'
                    }
                }
        
        return {'is_anomaly': False}
    
    def _analyze_risk_patterns(self, bank_tx: Transaction, company_entry: CompanyFinancial) -> Dict[str, Any]:
        """Analyze patterns that might indicate risk or fraud"""
        risk_factors = []
        risk_score = 0.0
        
        # High-value transaction risk
        amount = abs(company_entry.amount)
        if amount > 10000:  # > R$ 10,000
            risk_factors.append('high_value_transaction')
            risk_score += 0.3
        elif amount > 5000:  # > R$ 5,000
            risk_factors.append('medium_value_transaction')
            risk_score += 0.1
        
        # Unusual transaction type combinations
        if bank_tx.transaction_type != 'credit' and company_entry.transaction_type == 'income':
            risk_factors.append('transaction_type_mismatch_income')
            risk_score += 0.4
        elif bank_tx.transaction_type != 'debit' and company_entry.transaction_type == 'expense':
            risk_factors.append('transaction_type_mismatch_expense')
            risk_score += 0.4
        
        # Weekend/holiday transactions (potentially suspicious)
        if bank_tx.date:
            if bank_tx.date.weekday() >= 5:  # Weekend
                risk_factors.append('weekend_transaction')
                risk_score += 0.2
        
        # Round amount transactions (potentially suspicious)
        if amount > 1000 and amount % 1000 == 0:
            risk_factors.append('round_amount_transaction')
            risk_score += 0.1
        
        if risk_score > 0:
            severity = 'high' if risk_score > 0.5 else 'medium' if risk_score > 0.3 else 'low'
            
            return {
                'is_anomaly': True,
                'severity': severity,
                'score': min(risk_score, 0.9),
                'reason': f'Risk pattern detected with {len(risk_factors)} factors',
                'details': {
                    'risk_factors': risk_factors,
                    'total_risk_score': risk_score
                },
                'risk_factors': risk_factors
            }
        
        return {'is_anomaly': False}
    
    def _detect_duplicate_match(self, bank_tx: Transaction, company_entry: CompanyFinancial) -> Dict[str, Any]:
        """Detect potential duplicate matches"""
        # Check for existing reconciliation records with similar characteristics
        existing_matches = ReconciliationRecord.query.filter(
            ReconciliationRecord.bank_transaction_id != bank_tx.id,
            ReconciliationRecord.company_entry_id != company_entry.id,
            db.or_(
                ReconciliationRecord.status == 'pending',
                ReconciliationRecord.status == 'confirmed'
            )
        ).all()
        
        potential_duplicates = []
        
        for record in existing_matches:
            # Check for amount similarity
            if abs(record.bank_transaction.amount - bank_tx.amount) < 0.01:
                # Check for date proximity
                if record.bank_transaction.date and bank_tx.date:
                    date_diff = abs((record.bank_transaction.date - bank_tx.date).days)
                    if date_diff <= 7:  # Within 7 days
                        potential_duplicates.append({
                            'record_id': record.id,
                            'date_diff': date_diff,
                            'amount_diff': 0
                        })
        
        if potential_duplicates:
            return {
                'is_anomaly': True,
                'severity': 'medium',
                'score': 0.6,
                'reason': f'Potential duplicate match detected',
                'details': {
                    'potential_duplicates': potential_duplicates,
                    'duplicate_count': len(potential_duplicates)
                }
            }
        
        return {'is_anomaly': False}
    
    def _calculate_overall_severity(self, total_score: float, anomaly_count: int) -> str:
        """Calculate overall severity based on score and count"""
        if total_score >= self.severity_thresholds['critical'] or anomaly_count >= 3:
            return 'critical'
        elif total_score >= self.severity_thresholds['high']:
            return 'high'
        elif total_score >= self.severity_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confidence_level(self, match_score: float, anomaly_score: float, anomalies: List[Dict]) -> str:
        """Calculate confidence level in the anomaly assessment"""
        if not anomalies:
            return 'high'
        
        # High match score with anomalies increases confidence
        if match_score > 0.8 and anomaly_score > 0.6:
            return 'high'
        elif match_score > 0.6 and anomaly_score > 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _get_severity_weight(self, severity: str) -> float:
        """Get weight for severity level"""
        weights = {'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 1.0}
        return weights.get(severity, 0.5)
    
    def _generate_recommendations(self, anomalies: List[Dict], severity: str) -> List[str]:
        """Generate human-readable recommendations for anomaly resolution"""
        recommendations = []
        
        for anomaly in anomalies:
            anomaly_type = anomaly['type']
            anomaly_severity = anomaly['severity']
            
            if anomaly_type == 'amount_mismatch':
                if anomaly_severity in ['critical', 'high']:
                    recommendations.append("Verificar se há erro de digitação nos valores ou se há múltiplas transações relacionadas")
                else:
                    recommendations.append("Confirmar se a diferença de valor é aceitável para o negócio")
            
            elif anomaly_type == 'date_discrepancy':
                recommendations.append("Investigar o motivo da diferença de datas entre os registros")
            
            elif anomaly_type == 'description_mismatch':
                recommendations.append("Verificar se as descrições se referem à mesma transação, apesar das diferenças")
            
            elif anomaly_type == 'category_conflict':
                recommendations.append("Revisar a categorização para garantir consistência entre os sistemas")
            
            elif anomaly_type == 'unusual_pattern':
                recommendations.append("Investigar o padrão incomum para possível fraude ou erro de sistema")
            
            elif anomaly_type == 'duplicate_match':
                recommendations.append("Verificar se esta não é uma duplicata de outra reconciliação existente")
        
        # Severity-specific recommendations
        if severity == 'critical':
            recommendations.append("REVISÃO URGENTE: Esta reconciliação requer atenção imediata")
        elif severity == 'high':
            recommendations.append("Recomenda-se revisão por supervisor antes da confirmação")
        
        return list(set(recommendations))
    
    def _requires_human_review(self, anomalies: List[Dict], severity: str) -> bool:
        """Determine if human review is required"""
        if severity in ['critical', 'high']:
            return True
        
        # Multiple anomalies require human review
        if len(anomalies) >= 2:
            return True
        
        # Specific anomaly types require human review
        critical_anomaly_types = ['amount_mismatch', 'duplicate_match', 'compliance_risk']
        for anomaly in anomalies:
            if anomaly['type'] in critical_anomaly_types:
                return True
        
        return False
    
    def _calculate_priority_score(self, total_score: float, severity: str, anomalies: List[Dict]) -> float:
        """Calculate priority score for anomaly handling"""
        priority = total_score
        
        # Boost priority based on severity
        severity_multiplier = {'low': 1.0, 'medium': 1.5, 'high': 2.0, 'critical': 3.0}
        priority *= severity_multiplier.get(severity, 1.0)
        
        # Boost priority for multiple anomalies
        priority *= (1 + len(anomalies) * 0.2)
        
        return min(priority, 10.0)  # Cap at 10.0
    
    def get_anomaly_statistics(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics about anomalies for dashboard and reporting"""
        query = ReconciliationRecord.query.filter(ReconciliationRecord.is_anomaly == True)
        
        if user_id:
            query = query.filter(ReconciliationRecord.anomaly_reviewed_by == user_id)
        
        anomaly_records = query.all()
        
        stats = {
            'total_anomalies': len(anomaly_records),
            'by_severity': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'by_type': {},
            'by_status': {'pending': 0, 'confirmed': 0, 'dismissed': 0, 'escalated': 0},
            'by_action': {},
            'avg_resolution_time_hours': 0,
            'oldest_pending_anomaly': None,
            'most_common_anomaly_types': []
        }
        
        resolution_times = []
        
        for record in anomaly_records:
            # Count by severity
            if record.anomaly_severity in stats['by_severity']:
                stats['by_severity'][record.anomaly_severity] += 1
            
            # Count by type
            if record.anomaly_type:
                stats['by_type'][record.anomaly_type] = stats['by_type'].get(record.anomaly_type, 0) + 1
            
            # Count by action
            if record.anomaly_action:
                stats['by_action'][record.anomaly_action] = stats['by_action'].get(record.anomaly_action, 0) + 1
            
            # Calculate resolution time
            if record.anomaly_reviewed_at and record.anomaly_detected_at:
                resolution_time = (record.anomaly_reviewed_at - record.anomaly_detected_at).total_seconds() / 3600
                resolution_times.append(resolution_time)
            
            # Find oldest pending anomaly
            if record.anomaly_action in ['pending', None]:
                if not stats['oldest_pending_anomaly'] or \
                   record.anomaly_detected_at < stats['oldest_pending_anomaly']:
                    stats['oldest_pending_anomaly'] = record.anomaly_detected_at
        
        # Calculate average resolution time
        if resolution_times:
            stats['avg_resolution_time_hours'] = sum(resolution_times) / len(resolution_times)
        
        # Get most common anomaly types
        sorted_types = sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True)
        stats['most_common_anomaly_types'] = sorted_types[:5]
        
        return stats