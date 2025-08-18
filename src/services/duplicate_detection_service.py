import hashlib
from src.models.transaction import Transaction, UploadHistory
from src.models.company_financial import CompanyFinancial
from src.models.user import db


class DuplicateDetectionService:
    """
    Service for detecting duplicates at both file and entry levels
    """
    
    @staticmethod
    def calculate_file_hash(file_path):
        """
        Calculate SHA-256 hash of a file
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read the file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            raise Exception(f"Error calculating file hash: {str(e)}")
    
    @staticmethod
    def check_file_duplicate(file_hash):
        """
        Check if a file with the same hash was already processed
        Returns the existing UploadHistory record if found, None otherwise
        """
        try:
            existing_record = UploadHistory.query.filter_by(file_hash=file_hash).first()
            return existing_record
        except Exception as e:
            raise Exception(f"Error checking file duplicate: {str(e)}")
    
    @staticmethod
    def check_transaction_duplicate(account_id, date, amount, description):
        """
        Check if a transaction with the same details already exists
        """
        try:
            existing_transaction = Transaction.query.filter_by(
                account_id=account_id,
                date=date,
                amount=amount,
                description=description
            ).first()
            return existing_transaction is not None
        except Exception as e:
            raise Exception(f"Error checking transaction duplicate: {str(e)}")
    
    @staticmethod
    def check_financial_entry_duplicate(date, amount, description):
        """
        Check if a financial entry with the same details already exists
        """
        try:
            existing_entry = CompanyFinancial.query.filter_by(
                date=date,
                amount=amount,
                description=description
            ).first()
            return existing_entry is not None
        except Exception as e:
            raise Exception(f"Error checking financial entry duplicate: {str(e)}")
    
    @staticmethod
    def get_duplicate_transactions(account_id, date, amount, description):
        """
        Get all duplicate transactions with the same details
        """
        try:
            duplicate_transactions = Transaction.query.filter_by(
                account_id=account_id,
                date=date,
                amount=amount,
                description=description
            ).all()
            return duplicate_transactions
        except Exception as e:
            raise Exception(f"Error getting duplicate transactions: {str(e)}")
    
    @staticmethod
    def get_duplicate_financial_entries(date, amount, description):
        """
        Get all duplicate financial entries with the same details
        """
        try:
            duplicate_entries = CompanyFinancial.query.filter_by(
                date=date,
                amount=amount,
                description=description
            ).all()
            return duplicate_entries
        except Exception as e:
            raise Exception(f"Error getting duplicate financial entries: {str(e)}")