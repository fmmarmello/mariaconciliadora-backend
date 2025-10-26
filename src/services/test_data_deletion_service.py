import os
from datetime import datetime, timedelta
from sqlalchemy import inspect, text
from src.models.user import User, db
from src.models.transaction import ReconciliationRecord, Transaction, UploadHistory
from src.models.company_financial import CompanyFinancial
from src.models.user_preferences import (
    UserReconciliationConfig,
    UserReconciliationFeedback,
)
from src.utils.logging_config import get_logger, get_audit_logger

# Initialize loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()

class TestDataDeletionService:
    """
    Service for deleting test data with a triple check mechanism
    """
    
    def __init__(self):
        # Check if test data deletion is enabled
        self.enabled = os.getenv('ENABLE_TEST_DATA_DELETION', 'false').lower() == 'true'
    
    def identify_test_data(self, days_old=30, include_recent=False, preserve_user_ids=None,
                           preserve_org_ids=None, clear_users=False):
        """
        Identify test data based on creation date and other criteria
        
        Args:
            days_old (int): Number of days old data should be to be considered test data
            
        Returns:
            dict: Dictionary containing counts of test data in each table
        """
        if not self.enabled:
            logger.warning("Test data deletion feature is disabled")
            return {}
        
        try:
            # Calculate the cutoff date for test data
            cutoff_date = None if include_recent else datetime.utcnow() - timedelta(days=days_old)
            preserve_user_ids = preserve_user_ids or []
            preserve_org_ids = preserve_org_ids or []
            
            def _count(query, date_column=None):
                if cutoff_date and date_column is not None:
                    query = query.filter(date_column < cutoff_date)
                return query.count()
            
            test_transactions = _count(Transaction.query, Transaction.created_at)
            test_financial_entries = _count(CompanyFinancial.query, CompanyFinancial.created_at)
            test_upload_history = _count(UploadHistory.query, UploadHistory.upload_date)
            test_reconciliation = _count(ReconciliationRecord.query, ReconciliationRecord.created_at)
            test_feedback = _count(UserReconciliationFeedback.query, UserReconciliationFeedback.created_at)
            test_user_configs = _count(UserReconciliationConfig.query, UserReconciliationConfig.created_at)
            
            user_query = User.query
            if preserve_user_ids:
                user_query = user_query.filter(~User.id.in_(preserve_user_ids))
            test_users = user_query.count() if clear_users else 0
            
            org_count = 0
            engine = db.engine
            inspector = inspect(engine)
            if clear_users and inspector.has_table('organization'):
                sql = "SELECT COUNT(*) FROM organization"
                params = {}
                if preserve_org_ids:
                    placeholders = ','.join(f":p{i}" for i in range(len(preserve_org_ids)))
                    sql += f" WHERE id NOT IN ({placeholders})"
                    params = {f"p{i}": value for i, value in enumerate(preserve_org_ids)}
                org_count = db.session.execute(text(sql), params).scalar()
            
            return {
                'transactions': test_transactions,
                'company_financial': test_financial_entries,
                'upload_history': test_upload_history,
                'reconciliation_records': test_reconciliation,
                'user_reconciliation_feedback': test_feedback,
                'user_reconciliation_config': test_user_configs,
                'users': test_users,
                'organizations': org_count,
                'cutoff_date': cutoff_date,
                'total_count': (
                    test_transactions + test_financial_entries + test_upload_history +
                    test_reconciliation + test_feedback + test_user_configs +
                    test_users + org_count
                )
            }
        except Exception as e:
            logger.error(f"Error identifying test data: {str(e)}")
            raise Exception(f"Error identifying test data: {str(e)}")
    
    def get_test_data_details(self, days_old=30, include_recent=False, preserve_user_ids=None,
                               preserve_org_ids=None, clear_users=False):
        """
        Get detailed information about test data
        
        Args:
            days_old (int): Number of days old data should be to be considered test data
            
        Returns:
            dict: Dictionary containing detailed test data from each table
        """
        if not self.enabled:
            logger.warning("Test data deletion feature is disabled")
            return {}
        
        try:
            # Calculate the cutoff date for test data
            cutoff_date = None if include_recent else datetime.utcnow() - timedelta(days=days_old)
            preserve_user_ids = preserve_user_ids or []
            preserve_org_ids = preserve_org_ids or []
            
            def _items(query, date_column=None):
                if cutoff_date and date_column is not None:
                    query = query.filter(date_column < cutoff_date)
                return query.all()
            
            test_transactions = _items(Transaction.query, Transaction.created_at)
            test_financial_entries = _items(CompanyFinancial.query, CompanyFinancial.created_at)
            test_upload_history = _items(UploadHistory.query, UploadHistory.upload_date)
            test_reconciliation = _items(ReconciliationRecord.query, ReconciliationRecord.created_at)
            test_feedback = _items(UserReconciliationFeedback.query, UserReconciliationFeedback.created_at)
            test_user_configs = _items(UserReconciliationConfig.query, UserReconciliationConfig.created_at)
            
            user_query = User.query
            if preserve_user_ids:
                user_query = user_query.filter(~User.id.in_(preserve_user_ids))
            user_rows = user_query.all() if clear_users else []
            
            org_rows = []
            engine = db.engine
            inspector = inspect(engine)
            if clear_users and inspector.has_table('organization'):
                sql = "SELECT * FROM organization"
                params = {}
                if preserve_org_ids:
                    placeholders = ','.join(f":p{i}" for i in range(len(preserve_org_ids)))
                    sql += f" WHERE id NOT IN ({placeholders})"
                    params = {f"p{i}": value for i, value in enumerate(preserve_org_ids)}
                org_rows = db.session.execute(text(sql), params).fetchall()
            
            return {
                'transactions': [t.to_dict() for t in test_transactions],
                'company_financial': [f.to_dict() for f in test_financial_entries],
                'upload_history': [u.to_dict() for u in test_upload_history],
                'reconciliation_records': [r.to_dict() for r in test_reconciliation],
                'user_reconciliation_feedback': [f.to_dict() for f in test_feedback],
                'user_reconciliation_config': [c.to_dict() for c in test_user_configs],
                'users': [u.to_dict() for u in user_rows],
                'organizations': [dict(row._mapping) for row in org_rows],
                'cutoff_date': cutoff_date,
                'total_count': (
                    len(test_transactions) + len(test_financial_entries) + len(test_upload_history) +
                    len(test_reconciliation) + len(test_feedback) + len(test_user_configs) +
                    len(user_rows) + len(org_rows)
                )
            }
        except Exception as e:
            logger.error(f"Error getting test data details: {str(e)}")
            raise Exception(f"Error getting test data details: {str(e)}")
    
    def triple_check_delete(self, days_old=30, force=False, include_recent=False,
                             preserve_user_ids=None, preserve_org_ids=None, clear_users=False):
        """
        Triple check mechanism for deleting test data
        
        Args:
            days_old (int): Number of days old data should be to be considered test data
            force (bool): If True, skip confirmations and delete immediately
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Test data deletion feature is disabled")
            return False
        
        try:
            # First confirmation: Identify and list the data that would be deleted
            if not force:
                logger.info("=== TRIPLE CHECK DELETE - STEP 1 ===")
                print("=== TRIPLE CHECK DELETE - STEP 1 ===")
                test_data_summary = self.identify_test_data(
                    days_old,
                    include_recent=include_recent,
                    preserve_user_ids=preserve_user_ids,
                    preserve_org_ids=preserve_org_ids,
                    clear_users=clear_users
                )
                total_count = test_data_summary.get('total_count', 0)
                if total_count == 0:
                    logger.info("No test data found for deletion.")
                    print("No test data found for deletion.")
                    return False
                
                message = f"Found the following test data (older than {days_old} days):"
                details = [
                    f"  - Transactions: {test_data_summary.get('transactions', 0)}",
                    f"  - Company Financial Entries: {test_data_summary.get('company_financial', 0)}",
                    f"  - Upload History Records: {test_data_summary.get('upload_history', 0)}",
                    f"  - Total Records: {total_count}"
                ]
                
                logger.info(message)
                print(message)
                for detail in details:
                    logger.info(detail)
                    print(detail)
                
                confirm1 = input("Do you want to proceed with listing the detailed data? (yes/no): ")
                if confirm1.lower() != 'yes':
                    logger.info("Deletion cancelled at step 1.")
                    print("Deletion cancelled at step 1.")
                    return False
            
            # Second confirmation: Confirm the deletion action with a warning
            if not force:
                logger.info("=== TRIPLE CHECK DELETE - STEP 2 ===")
                print("\n=== TRIPLE CHECK DELETE - STEP 2 ===")
                test_data_details = self.get_test_data_details(
                    days_old,
                    include_recent=include_recent,
                    preserve_user_ids=preserve_user_ids,
                    preserve_org_ids=preserve_org_ids,
                    clear_users=clear_users
                )
                
                details_message = "Detailed test data to be deleted:"
                details = [
                    f"  - Transactions: {len(test_data_details.get('transactions', []))}",
                    f"  - Company Financial Entries: {len(test_data_details.get('company_financial', []))}",
                    f"  - Upload History Records: {len(test_data_details.get('upload_history', []))}"
                ]
                
                logger.info(details_message)
                print(details_message)
                for detail in details:
                    logger.info(detail)
                    print(detail)
                
                warning_msg = "WARNING: This action will permanently delete the data listed above. This operation cannot be undone."
                logger.warning(warning_msg)
                print(f"\n{warning_msg}")
                
                confirm2 = input("Do you want to proceed with the deletion? (yes/no): ")
                if confirm2.lower() != 'yes':
                    logger.info("Deletion cancelled at step 2.")
                    print("Deletion cancelled at step 2.")
                    return False
            
            # Third confirmation: Final confirmation before actual deletion
            if not force:
                logger.info("=== TRIPLE CHECK DELETE - STEP 3 ===")
                print("\n=== TRIPLE CHECK DELETE - STEP 3 ===")
                final_warning = "This is your final confirmation."
                logger.warning(final_warning)
                print(final_warning)
                confirm3 = input("Do you really want to permanently delete all test data? Type 'DELETE' to confirm: ")
                if confirm3 != 'DELETE':
                    logger.info("Deletion cancelled at step 3.")
                    print("Deletion cancelled at step 3.")
                    return False
            
            # Perform the actual deletion
            logger.info("Starting test data deletion process")
            print("\nDeleting test data...")
            result = self._delete_test_data(
                days_old,
                include_recent=include_recent,
                preserve_user_ids=preserve_user_ids,
                preserve_org_ids=preserve_org_ids,
                clear_users=clear_users
            )
            
            if result:
                success_msg = "Test data deletion completed successfully."
                logger.info(success_msg)
                print(success_msg)
            else:
                error_msg = "Test data deletion failed."
                logger.error(error_msg)
                print(error_msg)
            
            return result
            
        except Exception as e:
            error_msg = f"Error during triple check delete: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"Error during deletion: {str(e)}")
            return False
    
    def _delete_test_data(self, days_old=30, include_recent=False, preserve_user_ids=None,
                          preserve_org_ids=None, clear_users=False):
        """
        Actually delete test data from all relevant tables
        
        Args:
            days_old (int): Number of days old data should be to be considered test data
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Test data deletion feature is disabled")
            return False
        
        try:
            # Calculate the cutoff date for test data
            cutoff_date = None if include_recent else datetime.utcnow() - timedelta(days=days_old)
            preserve_user_ids = preserve_user_ids or []
            preserve_org_ids = preserve_org_ids or []
            
            def _filter(query, column=None):
                if cutoff_date and column is not None:
                    query = query.filter(column < cutoff_date)
                return query
            
            # Start a database transaction
            with db.session.begin():
                # Delete test data from transactions table
                deleted_feedback = _filter(
                    UserReconciliationFeedback.query, UserReconciliationFeedback.created_at
                ).delete(synchronize_session=False)
                
                deleted_configs = _filter(
                    UserReconciliationConfig.query, UserReconciliationConfig.created_at
                ).delete(synchronize_session=False)
                
                deleted_reconciliation = _filter(
                    ReconciliationRecord.query, ReconciliationRecord.created_at
                ).delete(synchronize_session=False)
                
                deleted_transactions = _filter(
                    Transaction.query, Transaction.created_at
                ).delete(synchronize_session=False)
                
                deleted_financial = _filter(
                    CompanyFinancial.query, CompanyFinancial.created_at
                ).delete(synchronize_session=False)
                
                deleted_upload_history = _filter(
                    UploadHistory.query, UploadHistory.upload_date
                ).delete(synchronize_session=False)
                
                deleted_users = 0
                if clear_users:
                    user_query = User.query
                    if preserve_user_ids:
                        user_query = user_query.filter(~User.id.in_(preserve_user_ids))
                    deleted_users = user_query.delete(synchronize_session=False)
                
                deleted_orgs = 0
                if clear_users:
                    engine = db.engine
                    inspector = inspect(engine)
                    if inspector.has_table('organization'):
                        sql = "DELETE FROM organization"
                        params = {}
                        if preserve_org_ids:
                            placeholders = ','.join(f":p{i}" for i in range(len(preserve_org_ids)))
                            sql += f" WHERE id NOT IN ({placeholders})"
                            params = {f"p{i}": value for i, value in enumerate(preserve_org_ids)}
                        deleted_orgs = db.session.execute(text(sql), params).rowcount
                
                # Log the deletion
                deletion_summary = (
                    "Deleted "
                    f"{deleted_transactions} transactions, "
                    f"{deleted_financial} financial entries, "
                    f"{deleted_upload_history} upload history records, "
                    f"{deleted_reconciliation} reconciliation records, "
                    f"{deleted_feedback} reconciliation feedback entries, "
                    f"{deleted_configs} reconciliation configurations, "
                    f"{deleted_users} users, "
                    f"{deleted_orgs} organizations"
                )
                logger.info(deletion_summary)
                
                # Audit log the deletion
                audit_logger.log_database_operation('delete', 'transactions', deleted_transactions, True)
                audit_logger.log_database_operation('delete', 'company_financial', deleted_financial, True)
                audit_logger.log_database_operation('delete', 'upload_history', deleted_upload_history, True)
                audit_logger.log_database_operation('delete', 'reconciliation_records', deleted_reconciliation, True)
                audit_logger.log_database_operation('delete', 'user_reconciliation_feedback', deleted_feedback, True)
                audit_logger.log_database_operation('delete', 'user_reconciliation_config', deleted_configs, True)
                if clear_users:
                    audit_logger.log_database_operation('delete', 'user', deleted_users, True)
                    if preserve_org_ids is not None:
                        audit_logger.log_database_operation('delete', 'organization', deleted_orgs, True)
                
                print(f"Deleted {deleted_transactions} transactions")
                print(f"Deleted {deleted_financial} company financial entries")
                print(f"Deleted {deleted_upload_history} upload history records")
                print(f"Deleted {deleted_reconciliation} reconciliation records")
                print(f"Deleted {deleted_feedback} reconciliation feedback entries")
                print(f"Deleted {deleted_configs} reconciliation configurations")
                if clear_users:
                    print(f"Deleted {deleted_users} users (preserved IDs: {preserve_user_ids})")
                    print(f"Deleted {deleted_orgs} organizations (preserved IDs: {preserve_org_ids})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting test data: {str(e)}")
            raise Exception(f"Error deleting test data: {str(e)}")
    
    def reset_all_data(self, preserve_user_ids=None, preserve_org_ids=None):
        """
        Convenience helper to wipe all dynamic data, keeping default entities.
        """
        return self._delete_test_data(
            days_old=0,
            include_recent=True,
            preserve_user_ids=preserve_user_ids or [1],
            preserve_org_ids=preserve_org_ids or [1],
            clear_users=True
        )

# Create a global instance of the service
test_data_deletion_service = TestDataDeletionService()
