import pandas as pd
from datetime import datetime, timedelta
from src.models.transaction import Transaction, ReconciliationRecord
from src.models.company_financial import CompanyFinancial
from src.models.user import db
from typing import List, Dict, Any, Optional
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.exceptions import (
    ReconciliationError, DatabaseError, RecordNotFoundError,
    InsufficientDataError, ValidationError
)
from src.utils.error_handler import (
    handle_service_errors, with_database_transaction, with_timeout
)

# Initialize loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()

class ReconciliationService:
    """
    Serviço para reconciliação entre transações bancárias e entradas financeiras da empresa
    """
    
    def __init__(self):
        pass
    
    @handle_service_errors('reconciliation_service')
    @with_timeout(300)  # 5 minute timeout for matching
    def find_matches(self, bank_transactions: List[Transaction], company_entries: List[CompanyFinancial]) -> List[Dict[str, Any]]:
        """
        Encontra correspondências entre transações bancárias e entradas financeiras da empresa
        """
        # Validate input data
        if not bank_transactions:
            raise InsufficientDataError('reconciliation matching', 1, 0)
        
        if not company_entries:
            raise InsufficientDataError('reconciliation matching', 1, 0)
        
        logger.info(f"Starting reconciliation matching. Bank transactions: {len(bank_transactions)}, Company entries: {len(company_entries)}")
        
        try:
            matches = []
            processed_transactions = 0
            matching_errors = 0
            
            # Validate transaction data before processing
            valid_bank_transactions = self._validate_transactions_for_matching(bank_transactions)
            valid_company_entries = self._validate_entries_for_matching(company_entries)
            
            if not valid_bank_transactions or not valid_company_entries:
                raise InsufficientDataError('reconciliation matching', 1, 0)
            
            logger.info(f"Validated data: {len(valid_bank_transactions)} bank transactions, {len(valid_company_entries)} company entries")
            
            for bank_transaction in valid_bank_transactions:
                try:
                    best_match = None
                    best_score = 0.0
                    processed_transactions += 1
                    
                    for company_entry in valid_company_entries:
                        try:
                            score = self._calculate_match_score(bank_transaction, company_entry)
                            
                            if score > best_score and score >= 0.7:  # Limiar mínimo para considerar match
                                best_score = score
                                best_match = company_entry
                                
                        except Exception as e:
                            logger.debug(f"Error calculating match score: {str(e)}")
                            continue
                    
                    if best_match:
                        match_data = {
                            'bank_transaction': bank_transaction,
                            'company_entry': best_match,
                            'match_score': best_score
                        }
                        
                        # Validate match before adding
                        if self._validate_match(match_data):
                            matches.append(match_data)
                            logger.debug(f"Match found for transaction {bank_transaction.id} with score {best_score:.2f}")
                        else:
                            logger.debug(f"Match validation failed for transaction {bank_transaction.id}")
                    
                except Exception as e:
                    logger.warning(f"Error processing bank transaction {bank_transaction.id}: {str(e)}")
                    matching_errors += 1
                    continue
            
            logger.info(f"Reconciliation matching completed. Processed: {processed_transactions}, "
                       f"Matches found: {len(matches)}, Errors: {matching_errors}")
            
            if matching_errors > 0:
                logger.warning(f"Encountered {matching_errors} errors during matching process")
            
            return matches
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, ValidationError)):
                raise
            
            logger.error(f"Critical error in reconciliation matching: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Matching process failed: {str(e)}")
    
    def _validate_transactions_for_matching(self, transactions: List[Transaction]) -> List[Transaction]:
        """Validate bank transactions for matching process."""
        valid_transactions = []
        
        for transaction in transactions:
            try:
                # Check required fields
                if not all([
                    transaction.id,
                    transaction.date,
                    transaction.amount is not None,
                    transaction.description
                ]):
                    logger.debug(f"Transaction {transaction.id} missing required fields")
                    continue
                
                # Check data validity
                if abs(transaction.amount) < 0.01:  # Skip very small amounts
                    logger.debug(f"Transaction {transaction.id} has very small amount: {transaction.amount}")
                    continue
                
                valid_transactions.append(transaction)
                
            except Exception as e:
                logger.debug(f"Error validating transaction {getattr(transaction, 'id', 'unknown')}: {str(e)}")
                continue
        
        logger.debug(f"Transaction validation: {len(valid_transactions)}/{len(transactions)} valid")
        return valid_transactions
    
    def _validate_entries_for_matching(self, entries: List[CompanyFinancial]) -> List[CompanyFinancial]:
        """Validate company entries for matching process."""
        valid_entries = []
        
        for entry in entries:
            try:
                # Check required fields
                if not all([
                    entry.id,
                    entry.date,
                    entry.amount is not None,
                    entry.description
                ]):
                    logger.debug(f"Entry {entry.id} missing required fields")
                    continue
                
                # Check data validity
                if abs(entry.amount) < 0.01:  # Skip very small amounts
                    logger.debug(f"Entry {entry.id} has very small amount: {entry.amount}")
                    continue
                
                valid_entries.append(entry)
                
            except Exception as e:
                logger.debug(f"Error validating entry {getattr(entry, 'id', 'unknown')}: {str(e)}")
                continue
        
        logger.debug(f"Entry validation: {len(valid_entries)}/{len(entries)} valid")
        return valid_entries
    
    def _validate_match(self, match_data: Dict[str, Any]) -> bool:
        """Validate a potential match."""
        try:
            bank_transaction = match_data['bank_transaction']
            company_entry = match_data['company_entry']
            match_score = match_data['match_score']
            
            # Basic validation
            if not all([bank_transaction, company_entry, match_score >= 0.7]):
                return False
            
            # Check for reasonable date difference (within 30 days)
            date_diff = abs((bank_transaction.date - company_entry.date).days)
            if date_diff > 30:
                logger.debug(f"Match rejected: date difference too large ({date_diff} days)")
                return False
            
            # Check for reasonable amount difference (within 1%)
            amount_diff = abs(abs(bank_transaction.amount) - abs(company_entry.amount))
            amount_ratio = amount_diff / max(abs(bank_transaction.amount), abs(company_entry.amount))
            if amount_ratio > 0.01:  # More than 1% difference
                logger.debug(f"Match rejected: amount difference too large ({amount_ratio:.2%})")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating match: {str(e)}")
            return False
    
    def _calculate_match_score(self, bank_transaction: Transaction, company_entry: CompanyFinancial) -> float:
        """
        Calcula o score de correspondência entre uma transação bancária e uma entrada financeira
        """
        score = 0.0
        
        # Verifica se os valores são iguais ou muito próximos
        if abs(abs(bank_transaction.amount) - abs(company_entry.amount)) < 0.01:
            score += 0.4
        elif abs(abs(bank_transaction.amount) - abs(company_entry.amount)) < 1.0:
            score += 0.2
        
        # Verifica se as datas são próximas (até 3 dias de diferença)
        if bank_transaction.date and company_entry.date:
            date_diff = abs((bank_transaction.date - company_entry.date).days)
            if date_diff == 0:
                score += 0.3
            elif date_diff <= 3:
                score += 0.15
        
        # Verifica similaridade nas descrições
        description_similarity = self._calculate_description_similarity(
            bank_transaction.description, 
            company_entry.description
        )
        score += description_similarity * 0.3
        
        return min(score, 1.0)  # Limita o score a 1.0
    
    def _calculate_description_similarity(self, bank_description: str, company_description: str) -> float:
        """
        Calcula a similaridade entre duas descrições usando uma abordagem simples
        """
        # Converte para minúsculas e remove espaços extras
        bank_desc = bank_description.lower().strip()
        company_desc = company_description.lower().strip()
        
        # Verifica se uma descrição está contida na outra
        if bank_desc in company_desc or company_desc in bank_desc:
            return 0.8
        
        # Verifica palavras em comum
        bank_words = set(bank_desc.split())
        company_words = set(company_desc.split())
        
        if len(bank_words) > 0 and len(company_words) > 0:
            common_words = bank_words.intersection(company_words)
            similarity = len(common_words) / max(len(bank_words), len(company_words))
            return similarity * 0.6
        
        return 0.0
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def create_reconciliation_records(self, matches: List[Dict[str, Any]]) -> List[ReconciliationRecord]:
        """
        Cria registros de reconciliação no banco de dados
        """
        if not matches:
            raise InsufficientDataError('reconciliation record creation', 1, 0)
        
        logger.info(f"Creating {len(matches)} reconciliation records")
        
        try:
            records = []
            creation_errors = 0
            
            for i, match in enumerate(matches):
                try:
                    # Validate match data
                    if not self._validate_match_for_record_creation(match):
                        logger.warning(f"Match {i+1} failed validation for record creation")
                        creation_errors += 1
                        continue
                    
                    # Check for existing reconciliation record
                    existing_record = ReconciliationRecord.query.filter_by(
                        bank_transaction_id=match['bank_transaction'].id,
                        company_entry_id=match['company_entry'].id
                    ).first()
                    
                    if existing_record:
                        logger.debug(f"Reconciliation record already exists for transaction {match['bank_transaction'].id}")
                        records.append(existing_record)
                        continue
                    
                    # Create new record
                    record = ReconciliationRecord(
                        bank_transaction_id=match['bank_transaction'].id,
                        company_entry_id=match['company_entry'].id,
                        match_score=match['match_score'],
                        status='pending'
                    )
                    
                    db.session.add(record)
                    records.append(record)
                    
                except Exception as e:
                    logger.error(f"Error creating reconciliation record for match {i+1}: {str(e)}")
                    creation_errors += 1
                    continue
            
            if not records:
                raise ReconciliationError("No valid reconciliation records could be created")
            
            # Commit handled by @with_database_transaction decorator
            
            logger.info(f"Successfully created {len(records)} reconciliation records (errors: {creation_errors})")
            audit_logger.log_database_operation('insert', 'reconciliation_records', len(records), True)
            
            return records
            
        except Exception as e:
            if isinstance(e, (InsufficientDataError, ReconciliationError)):
                raise
            
            logger.error(f"Critical error creating reconciliation records: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to create reconciliation records: {str(e)}")
    
    def _validate_match_for_record_creation(self, match: Dict[str, Any]) -> bool:
        """Validate match data for record creation."""
        try:
            required_keys = ['bank_transaction', 'company_entry', 'match_score']
            if not all(key in match for key in required_keys):
                return False
            
            bank_transaction = match['bank_transaction']
            company_entry = match['company_entry']
            match_score = match['match_score']
            
            # Validate objects have required attributes
            if not all([
                hasattr(bank_transaction, 'id') and bank_transaction.id,
                hasattr(company_entry, 'id') and company_entry.id,
                isinstance(match_score, (int, float)) and 0 <= match_score <= 1
            ]):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating match for record creation: {str(e)}")
            return False
    
    @handle_service_errors('reconciliation_service')
    def get_pending_reconciliations(self) -> List[ReconciliationRecord]:
        """
        Obtém todos os registros de reconciliação pendentes
        """
        try:
            records = ReconciliationRecord.query.filter_by(status='pending').all()
            logger.info(f"Retrieved {len(records)} pending reconciliation records")
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving pending reconciliations: {str(e)}", exc_info=True)
            raise DatabaseError("Failed to retrieve pending reconciliations")
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def confirm_reconciliation(self, reconciliation_id: int) -> bool:
        """
        Confirma uma reconciliação
        """
        if not reconciliation_id or reconciliation_id <= 0:
            raise ValidationError("Invalid reconciliation ID")
        
        logger.info(f"Confirming reconciliation record {reconciliation_id}")
        
        try:
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
            
            # Validate current status
            if record.status == 'confirmed':
                logger.info(f"Reconciliation {reconciliation_id} already confirmed")
                return True
            
            if record.status == 'rejected':
                logger.warning(f"Cannot confirm rejected reconciliation {reconciliation_id}")
                raise ReconciliationError("Cannot confirm a rejected reconciliation")
            
            # Update status
            record.status = 'confirmed'
            # Commit handled by @with_database_transaction decorator
            
            logger.info(f"Reconciliation {reconciliation_id} confirmed successfully")
            audit_logger.log_reconciliation_operation('confirm', 1, 'confirmed', {
                'reconciliation_id': reconciliation_id,
                'match_score': record.match_score
            })
            
            return True
            
        except Exception as e:
            if isinstance(e, (RecordNotFoundError, ValidationError, ReconciliationError)):
                raise
            
            logger.error(f"Error confirming reconciliation {reconciliation_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to confirm reconciliation: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    @with_database_transaction
    def reject_reconciliation(self, reconciliation_id: int) -> bool:
        """
        Rejeita uma reconciliação
        """
        if not reconciliation_id or reconciliation_id <= 0:
            raise ValidationError("Invalid reconciliation ID")
        
        logger.info(f"Rejecting reconciliation record {reconciliation_id}")
        
        try:
            record = ReconciliationRecord.query.get(reconciliation_id)
            if not record:
                raise RecordNotFoundError('ReconciliationRecord', reconciliation_id)
            
            # Validate current status
            if record.status == 'rejected':
                logger.info(f"Reconciliation {reconciliation_id} already rejected")
                return True
            
            if record.status == 'confirmed':
                logger.warning(f"Cannot reject confirmed reconciliation {reconciliation_id}")
                raise ReconciliationError("Cannot reject a confirmed reconciliation")
            
            # Update status
            record.status = 'rejected'
            # Commit handled by @with_database_transaction decorator
            
            logger.info(f"Reconciliation {reconciliation_id} rejected successfully")
            audit_logger.log_reconciliation_operation('reject', 1, 'rejected', {
                'reconciliation_id': reconciliation_id,
                'match_score': record.match_score
            })
            
            return True
            
        except Exception as e:
            if isinstance(e, (RecordNotFoundError, ValidationError, ReconciliationError)):
                raise
            
            logger.error(f"Error rejecting reconciliation {reconciliation_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to reject reconciliation: {str(e)}")
    
    @handle_service_errors('reconciliation_service')
    def get_reconciliation_report(self) -> Dict[str, Any]:
        """
        Gera um relatório de reconciliação
        """
        logger.info("Generating reconciliation report")
        
        try:
            # Get basic counts
            total_records = ReconciliationRecord.query.count()
            confirmed_records = ReconciliationRecord.query.filter_by(status='confirmed').count()
            pending_records = ReconciliationRecord.query.filter_by(status='pending').count()
            rejected_records = ReconciliationRecord.query.filter_by(status='rejected').count()
            
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics()
            
            # Calculate reconciliation rate
            reconciliation_rate = confirmed_records / total_records if total_records > 0 else 0
            
            # Get additional statistics
            avg_match_score = self._calculate_average_match_score()
            recent_activity = self._get_recent_reconciliation_activity()
            
            report = {
                'summary': {
                    'total_records': total_records,
                    'confirmed': confirmed_records,
                    'pending': pending_records,
                    'rejected': rejected_records,
                    'reconciliation_rate': round(reconciliation_rate, 4),
                    'average_match_score': round(avg_match_score, 4) if avg_match_score else 0
                },
                'financials': financial_metrics,
                'recent_activity': recent_activity,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Reconciliation report generated. Total records: {total_records}, "
                       f"Confirmed: {confirmed_records}, Rate: {reconciliation_rate:.2%}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating reconciliation report: {str(e)}", exc_info=True)
            raise ReconciliationError(f"Failed to generate reconciliation report: {str(e)}")
    
    def _calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial metrics for reconciliation report."""
        try:
            confirmed_matches = ReconciliationRecord.query.filter_by(status='confirmed').all()
            
            total_reconciled_value = 0
            reconciled_credits = 0
            reconciled_debits = 0
            
            for record in confirmed_matches:
                try:
                    if record.bank_transaction and record.bank_transaction.amount is not None:
                        amount = record.bank_transaction.amount
                        total_reconciled_value += abs(amount)
                        
                        if amount > 0:
                            reconciled_credits += amount
                        else:
                            reconciled_debits += abs(amount)
                            
                except Exception as e:
                    logger.debug(f"Error processing record {record.id}: {str(e)}")
                    continue
            
            return {
                'total_reconciled_value': round(total_reconciled_value, 2),
                'reconciled_credits': round(reconciled_credits, 2),
                'reconciled_debits': round(reconciled_debits, 2),
                'net_reconciled_flow': round(reconciled_credits - reconciled_debits, 2)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating financial metrics: {str(e)}")
            return {
                'total_reconciled_value': 0,
                'reconciled_credits': 0,
                'reconciled_debits': 0,
                'net_reconciled_flow': 0
            }
    
    def _calculate_average_match_score(self) -> Optional[float]:
        """Calculate average match score for confirmed reconciliations."""
        try:
            confirmed_records = ReconciliationRecord.query.filter_by(status='confirmed').all()
            
            if not confirmed_records:
                return None
            
            total_score = sum(record.match_score for record in confirmed_records if record.match_score)
            return total_score / len(confirmed_records)
            
        except Exception as e:
            logger.debug(f"Error calculating average match score: {str(e)}")
            return None
    
    def _get_recent_reconciliation_activity(self) -> Dict[str, int]:
        """Get recent reconciliation activity (last 7 days)."""
        try:
            seven_days_ago = datetime.now() - timedelta(days=7)
            
            recent_confirmed = ReconciliationRecord.query.filter(
                ReconciliationRecord.status == 'confirmed',
                ReconciliationRecord.created_at >= seven_days_ago
            ).count()
            
            recent_rejected = ReconciliationRecord.query.filter(
                ReconciliationRecord.status == 'rejected',
                ReconciliationRecord.created_at >= seven_days_ago
            ).count()
            
            return {
                'confirmed_last_7_days': recent_confirmed,
                'rejected_last_7_days': recent_rejected,
                'total_activity_last_7_days': recent_confirmed + recent_rejected
            }
            
        except Exception as e:
            logger.debug(f"Error getting recent activity: {str(e)}")
            return {
                'confirmed_last_7_days': 0,
                'rejected_last_7_days': 0,
                'total_activity_last_7_days': 0
            }