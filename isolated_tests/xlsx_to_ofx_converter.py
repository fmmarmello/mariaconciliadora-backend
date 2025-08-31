#!/usr/bin/env python3
"""
XLSX to OFX Converter for Test Data Generation

This script converts XLSX financial data to OFX format for testing purposes.
Creates two OFX files: one exact match and one with slight differences.
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import re
import random

class XLSXToOFXConverter:
    """Converts XLSX financial data to OFX format"""

    def __init__(self):
        self.supported_columns = {
            'date': ['data', 'date', 'dia'],
            'description': ['descricao', 'description', 'histórico', 'historico'],
            'amount': ['valor', 'amount', 'value'],
            'category': ['categoria', 'category'],
            'type': ['tipo', 'type', 'transaction type']
        }

    def convert_xlsx_to_ofx(self, xlsx_path: str, output_exact: str, output_modified: str):
        """Convert XLSX file to two OFX files"""

        # Read and process XLSX data
        print(f"Reading XLSX file: {xlsx_path}")
        financial_data = self._parse_xlsx_file(xlsx_path)

        if not financial_data:
            raise ValueError("No valid data found in XLSX file")

        print(f"Processed {len(financial_data)} transactions from XLSX")

        # Generate exact OFX file
        self._generate_ofx_file(financial_data, output_exact, modify_data=False)
        print(f"Created exact OFX file: {output_exact}")

        # Generate modified OFX file with slight differences
        self._generate_ofx_file(financial_data, output_modified, modify_data=True)
        print(f"Created modified OFX file: {output_modified}")

    def _parse_xlsx_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse XLSX file and return structured financial data"""
        try:
            # Read XLSX file
            df = pd.read_excel(file_path)
            print(f"XLSX loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")

            # Normalize column names
            df.columns = [self._normalize_column_name(col) for col in df.columns]

            financial_data = []
            for index, row in df.iterrows():
                try:
                    entry = self._process_row(row)
                    if entry:
                        financial_data.append(entry)
                except Exception as e:
                    print(f"Warning: Error processing row {index + 1}: {str(e)}")
                    continue

            return financial_data

        except Exception as e:
            raise ValueError(f"Error reading XLSX file: {str(e)}")

    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize column names to standard format"""
        column_name = str(column_name).strip().lower()
        column_name = re.sub(r'[^\w\s]', '', column_name)
        return column_name

    def _process_row(self, row: pd.Series) -> Dict[str, Any]:
        """Process a single XLSX row"""
        try:
            entry = {
                'date': self._parse_date(row.get('data')),
                'description': str(row.get('description', '')).strip(),
                'amount': self._parse_amount(row.get('valor')),
                'category': str(row.get('tipo', '')),
            }

            # Validate entry has required fields
            if not entry['date'] or not entry['description'] or entry['amount'] is None:
                return None

            return entry

        except Exception as e:
            print(f"Error processing row: {str(e)}")
            return None

    def _parse_date(self, date_value) -> datetime.date:
        """Parse date from various formats"""
        if pd.isna(date_value) or date_value is None:
            return None

        if hasattr(date_value, 'date'):
            return date_value.date()

        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y',
            '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d'
        ]

        date_str = str(date_value).strip()

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        try:
            return pd.to_datetime(date_value, dayfirst=True).date()
        except Exception:
            print(f"Failed to parse date: {date_value}")
            return None

    def _parse_amount(self, amount_value) -> float:
        """Parse monetary amount"""
        if pd.isna(amount_value) or amount_value is None:
            return 0.0

        if isinstance(amount_value, (int, float)):
            return float(amount_value)

        amount_str = str(amount_value).strip()

        if not amount_str:
            return 0.0

        try:
            amount_str = re.sub(r'[R$\s]', '', amount_str)

            if ',' in amount_str and '.' in amount_str:
                amount_str = amount_str.replace('.', '').replace(',', '.')
            elif ',' in amount_str:
                comma_parts = amount_str.split(',')
                if len(comma_parts) == 2 and len(comma_parts[1]) <= 2:
                    amount_str = amount_str.replace(',', '.')
                else:
                    amount_str = amount_str.replace(',', '')

            if amount_str.startswith('(') and amount_str.endswith(')'):
                amount_str = '-' + amount_str[1:-1]

            return float(amount_str)

        except (ValueError, TypeError) as e:
            print(f"Failed to parse amount '{amount_value}': {str(e)}")
            return 0.0

    def _generate_ofx_file(self, transactions: List[Dict[str, Any]], output_path: str, modify_data: bool = False):
        """Generate OFX file from transaction data"""

        # Calculate date range
        dates = [t['date'] for t in transactions if t['date']]
        if dates:
            start_date = min(dates)
            end_date = max(dates)
        else:
            start_date = datetime.now().date()
            end_date = datetime.now().date()

        # OFX header
        ofx_content = self._generate_ofx_header()

        # Bank account information
        ofx_content += self._generate_bank_info()

        # Transaction list header
        ofx_content += self._generate_transaction_list_header(start_date, end_date)

        # Process transactions
        total_balance = 0.0
        fitid_counter = 1

        for transaction in transactions:
            if not transaction.get('date') or not transaction.get('description'):
                continue

            # Apply modifications if requested
            if modify_data:
                transaction = self._modify_transaction(transaction)

            # Generate transaction entry
            transaction_xml = self._generate_transaction_entry(transaction, fitid_counter)
            ofx_content += transaction_xml

            # Update balance
            amount = transaction.get('amount', 0)
            total_balance += amount
            fitid_counter += 1

        # Close transaction list
        ofx_content += "</BANKTRANLIST>\n"

        # Add ledger balance
        ofx_content += self._generate_ledger_balance(total_balance, end_date)

        # Close OFX structure
        ofx_content += self._generate_ofx_footer()

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ofx_content)

    def _generate_ofx_header(self) -> str:
        """Generate OFX file header"""
        return """OFXHEADER:100
DATA:OFXSGML
VERSION:102
SECURITY:NONE
ENCODING:USASCII
CHARSET:1252
COMPRESSION:NONE
OLDFILEUID:NONE
NEWFILEUID:NONE

<OFX>
<SIGNONMSGSRSV1>
<SONRS>
<STATUS>
<CODE>0
<SEVERITY>INFO
</STATUS>
<DTSERVER>{}</DATE>
<LANGUAGE>POR
</SONRS>
</SIGNONMSGSRSV1>
<BANKMSGSRSV1>
<STMTTRNRS>
<TRNUID>1
<STATUS>
<CODE>0
<SEVERITY>INFO
</STATUS>
<STMTRS>
<CURDEF>BRL
""".format(datetime.now().strftime('%Y%m%d%H%M%S'))

    def _generate_bank_info(self) -> str:
        """Generate bank account information"""
        return """<BANKACCTFROM>
<BANKID>104</BANKID>
<ACCTID>12345-6</ACCTID>
<ACCTTYPE>CHECKING</ACCTTYPE>
</BANKACCTFROM>
"""

    def _generate_transaction_list_header(self, start_date, end_date) -> str:
        """Generate transaction list header with date range"""
        return """<BANKTRANLIST>
<DTSTART>{}</DTSTART>
<DTEND>{}</DTEND>
""".format(
            start_date.strftime('%Y%m%d%H%M%S'),
            end_date.strftime('%Y%m%d%H%M%S')
        )

    def _generate_transaction_entry(self, transaction: Dict[str, Any], fitid: int) -> str:
        """Generate individual transaction entry"""

        # Determine transaction type
        amount = transaction.get('amount', 0)
        trn_type = "CREDIT" if amount > 0 else "DEBIT"

        # Format date
        date = transaction.get('date')
        if hasattr(date, 'strftime'):
            dt_posted = date.strftime('%Y%m%d%H%M%S')
        else:
            dt_posted = datetime.now().strftime('%Y%m%d%H%M%S')

        # Format amount (negative for debits)
        formatted_amount = f"{amount:.2f}"

        # Get description
        description = transaction.get('description', '').strip()
        if len(description) > 255:
            description = description[:255]

        return f"""<STMTTRN>
<TRNTYPE>{trn_type}</TRNTYPE>
<DTPOSTED>{dt_posted}</DTPOSTED>
<TRNAMT>{formatted_amount}</TRNAMT>
<FITID>{fitid:012d}</FITID>
<MEMO>{description}</MEMO>
</STMTTRN>
"""

    def _generate_ledger_balance(self, balance: float, as_of_date) -> str:
        """Generate ledger balance section"""
        return """<LEDGERBAL>
<BALAMT>{:.2f}</BALAMT>
<DTASOF>{}</DTASOF>
</LEDGERBAL>
""".format(balance, as_of_date.strftime('%Y%m%d%H%M%S'))

    def _generate_ofx_footer(self) -> str:
        """Generate OFX file footer"""
        return """</STMTRS>
</STMTTRNRS>
</BANKMSGSRSV1>
</OFX>
"""

    def _modify_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Apply slight modifications to transaction for testing"""
        modified = transaction.copy()

        # Modify amount by small percentage (0.5% to 2%)
        import random
        amount = modified.get('amount', 0)
        if amount != 0:
            variation = random.uniform(0.005, 0.02)  # 0.5% to 2%
            if random.choice([True, False]):
                modified['amount'] = amount * (1 + variation)
            else:
                modified['amount'] = amount * (1 - variation)

        # Occasionally modify description slightly
        if random.random() < 0.3:  # 30% chance
            description = modified.get('description', '')
            if 'LTDA' in description:
                modified['description'] = description.replace('LTDA', 'S.A.')
            elif 'S.A.' in description:
                modified['description'] = description.replace('S.A.', 'LTDA')
            elif len(description) > 10:
                # Add or remove a word occasionally
                words = description.split()
                if len(words) > 2 and random.random() < 0.5:
                    words.insert(random.randint(0, len(words)), 'TEST')
                    modified['description'] = ' '.join(words)

        return modified

def main():
    """Main function to run the converter"""

    # File paths
    xlsx_file = "mariaconciliadora-backend/samples/Novembro21.xlsx"
    exact_ofx = "tests/test_novembro21_exact.ofx"
    modified_ofx = "tests/test_novembro21_modified.ofx"

    # Create converter and process files
    converter = XLSXToOFXConverter()

    try:
        converter.convert_xlsx_to_ofx(xlsx_file, exact_ofx, modified_ofx)
        print("\n[SUCCESS] Successfully created OFX test files!")
        print(f"   Exact match: {exact_ofx}")
        print(f"   Modified: {modified_ofx}")

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()