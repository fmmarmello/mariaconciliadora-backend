import os
import logging
from datetime import datetime, timedelta
from src.models.user import db
from src.models.transaction import Transaction, UploadHistory
from src.models.company_financial import CompanyFinancial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataDeletionService:
    """
    Service for deleting test data with a triple check mechanism
    """
    
    def __init__(self):
        # Check if test data deletion is enabled
        self.enabled = os.getenv('ENABLE_TEST_DATA_DELETION', 'false').lower() == 'true'
    
    def identify_test_data(self, days_old=30):
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
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Count test data in each table
            test_transactions = Transaction.query.filter(
                Transaction.created_at < cutoff_date
            ).count()
            
            test_financial_entries = CompanyFinancial.query.filter(
                CompanyFinancial.created_at < cutoff_date
            ).count()
            
            test_upload_history = UploadHistory.query.filter(
                UploadHistory.upload_date < cutoff_date
            ).count()
            
            return {
                'transactions': test_transactions,
                'company_financial': test_financial_entries,
                'upload_history': test_upload_history,
                'cutoff_date': cutoff_date,
                'total_count': test_transactions + test_financial_entries + test_upload_history
            }
        except Exception as e:
            logger.error(f"Error identifying test data: {str(e)}")
            raise Exception(f"Error identifying test data: {str(e)}")
    
    def get_test_data_details(self, days_old=30):
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
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Get test data from each table
            test_transactions = Transaction.query.filter(
                Transaction.created_at < cutoff_date
            ).all()
            
            test_financial_entries = CompanyFinancial.query.filter(
                CompanyFinancial.created_at < cutoff_date
            ).all()
            
            test_upload_history = UploadHistory.query.filter(
                UploadHistory.upload_date < cutoff_date
            ).all()
            
            return {
                'transactions': [t.to_dict() for t in test_transactions],
                'company_financial': [f.to_dict() for f in test_financial_entries],
                'upload_history': [u.to_dict() for u in test_upload_history],
                'cutoff_date': cutoff_date,
                'total_count': len(test_transactions) + len(test_financial_entries) + len(test_upload_history)
            }
        except Exception as e:
            logger.error(f"Error getting test data details: {str(e)}")
            raise Exception(f"Error getting test data details: {str(e)}")
    
    def triple_check_delete(self, days_old=30, force=False):
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
                print("=== TRIPLE CHECK DELETE - STEP 1 ===")
                test_data_summary = self.identify_test_data(days_old)
                total_count = test_data_summary.get('total_count', 0)
                if total_count == 0:
                    print("No test data found for deletion.")
                    return False
                
                print(f"Found the following test data (older than {days_old} days):")
                print(f"  - Transactions: {test_data_summary.get('transactions', 0)}")
                print(f"  - Company Financial Entries: {test_data_summary.get('company_financial', 0)}")
                print(f"  - Upload History Records: {test_data_summary.get('upload_history', 0)}")
                print(f"  - Total Records: {total_count}")
                
                confirm1 = input("Do you want to proceed with listing the detailed data? (yes/no): ")
                if confirm1.lower() != 'yes':
                    print("Deletion cancelled at step 1.")
                    return False
            
            # Second confirmation: Confirm the deletion action with a warning
            if not force:
                print("\n=== TRIPLE CHECK DELETE - STEP 2 ===")
                test_data_details = self.get_test_data_details(days_old)
                print(f"Detailed test data to be deleted:")
                print(f"  - Transactions: {len(test_data_details.get('transactions', []))}")
                print(f"  - Company Financial Entries: {len(test_data_details.get('company_financial', []))}")
                print(f"  - Upload History Records: {len(test_data_details.get('upload_history', []))}")
                
                print("\nWARNING: This action will permanently delete the data listed above.")
                print("This operation cannot be undone.")
                
                confirm2 = input("Do you want to proceed with the deletion? (yes/no): ")
                if confirm2.lower() != 'yes':
                    print("Deletion cancelled at step 2.")
                    return False
            
            # Third confirmation: Final confirmation before actual deletion
            if not force:
                print("\n=== TRIPLE CHECK DELETE - STEP 3 ===")
                print("This is your final confirmation.")
                confirm3 = input("Do you really want to permanently delete all test data? Type 'DELETE' to confirm: ")
                if confirm3 != 'DELETE':
                    print("Deletion cancelled at step 3.")
                    return False
            
            # Perform the actual deletion
            print("\nDeleting test data...")
            result = self._delete_test_data(days_old)
            
            if result:
                print("Test data deletion completed successfully.")
                logger.info("Test data deletion completed successfully.")
            else:
                print("Test data deletion failed.")
                logger.error("Test data deletion failed.")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during triple check delete: {str(e)}")
            print(f"Error during deletion: {str(e)}")
            return False
    
    def _delete_test_data(self, days_old=30):
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
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Start a database transaction
            with db.session.begin():
                # Delete test data from transactions table
                deleted_transactions = Transaction.query.filter(
                    Transaction.created_at < cutoff_date
                ).delete()
                
                # Delete test data from company_financial table
                deleted_financial = CompanyFinancial.query.filter(
                    CompanyFinancial.created_at < cutoff_date
                ).delete()
                
                # Delete test data from upload_history table
                deleted_upload_history = UploadHistory.query.filter(
                    UploadHistory.upload_date < cutoff_date
                ).delete()
                
                # Log the deletion
                logger.info(f"Deleted {deleted_transactions} transactions, "
                           f"{deleted_financial} financial entries, "
                           f"{deleted_upload_history} upload history records")
                
                print(f"Deleted {deleted_transactions} transactions")
                print(f"Deleted {deleted_financial} company financial entries")
                print(f"Deleted {deleted_upload_history} upload history records")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting test data: {str(e)}")
            raise Exception(f"Error deleting test data: {str(e)}")

# Create a global instance of the service
test_data_deletion_service = TestDataDeletionService()