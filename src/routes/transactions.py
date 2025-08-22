from flask import Blueprint, request, jsonify, g
from werkzeug.utils import secure_filename
import os
import tempfile
from datetime import datetime, timedelta
from src.models.transaction import Transaction, UploadHistory, ReconciliationRecord, db
from src.models.company_financial import CompanyFinancial
from src.services.ofx_processor import OFXProcessor
from src.services.xlsx_processor import XLSXProcessor
from src.services.ai_service import AIService
from src.services.reconciliation_service import ReconciliationService
from src.services.duplicate_detection_service import DuplicateDetectionService
from src.utils.logging_config import get_logger, get_audit_logger
from src.utils.error_handler import handle_errors, with_resource_check
from src.utils.exceptions import (
    ValidationError, FileProcessingError, InvalidFileFormatError,
    FileSizeExceededError, DuplicateFileError, InsufficientDataError
)
from src.utils.validators import validate_pagination_params, validate_file_upload
from src.utils.validation_middleware import (
    validate_file_upload, validate_input_fields, validate_financial_data,
    rate_limit, require_content_type, sanitize_path_params
)

# Initialize loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()

transactions_bp = Blueprint('transactions', __name__)

# Configurações de upload
ALLOWED_EXTENSIONS = {'ofx', 'qfx'}
ALLOWED_XLSX_EXTENSIONS = {'xlsx'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_xlsx_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_XLSX_EXTENSIONS

@transactions_bp.route('/upload-ofx', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=50, window_minutes=60)  # Limit file uploads
@validate_file_upload(['ofx', 'qfx'], max_size_mb=16)
def upload_ofx():
    """
    Endpoint para upload e processamento de arquivos OFX
    """
    logger.info("OFX upload request received")
    
    # Validate request
    if 'file' not in request.files:
        logger.warning("Upload request without file")
        raise ValidationError("Nenhum arquivo enviado", user_message="Nenhum arquivo foi enviado.")
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Upload request with empty filename")
        raise ValidationError("Nenhum arquivo selecionado", user_message="Nenhum arquivo foi selecionado.")
    
    if not allowed_file(file.filename):
        logger.warning(f"Upload request with invalid file type: {file.filename}")
        raise InvalidFileFormatError(file.filename, ['ofx', 'qfx'])
        
    # Salva o arquivo temporariamente
    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    
    try:
        file.save(temp_path)
        
        # Validate file size
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE:
            raise FileSizeExceededError(filename, file_size, MAX_FILE_SIZE)
        
        logger.info(f"OFX file uploaded: {filename} ({file_size} bytes)")
        
        # Additional file validation
        validation_result = validate_file_upload(temp_path, filename, 'ofx')
        if not validation_result.is_valid:
            raise FileProcessingError(f"File validation failed: {', '.join(validation_result.errors)}", filename)
        
        # Verifica se o arquivo é duplicado
        file_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
        audit_logger.log_file_upload(filename, 'OFX', file_size=file_size, file_hash=file_hash)
        
        existing_file = DuplicateDetectionService.check_file_duplicate(file_hash)
        
        if existing_file:
            logger.warning(f"Duplicate file detected: {filename} (hash: {file_hash})")
            audit_logger.log_duplicate_detection('file_level', 1, {
                'filename': filename,
                'original_upload_date': existing_file.upload_date.isoformat() if existing_file.upload_date else None
            })
            
            raise DuplicateFileError(filename, existing_file.upload_date.isoformat() if existing_file.upload_date else None)
        
        # Processa o arquivo OFX
        processor = OFXProcessor()
        ai_service = AIService()
        
        logger.info(f"Starting OFX processing for file: {filename}")
        
        # Parse do OFX
        ofx_data = processor.parse_ofx_file(temp_path)
        logger.info(f"OFX parsed successfully. Bank: {ofx_data['bank_name']}, Transactions: {len(ofx_data['transactions'])}")
        
        # Valida as transações
        valid_transactions = processor.validate_transactions(ofx_data['transactions'])
        logger.info(f"Transaction validation completed. Valid transactions: {len(valid_transactions)}")
        
        if not valid_transactions:
            raise InsufficientDataError('OFX processing', 1, 0)
        
        # Detecta duplicatas usando o processor
        duplicates = processor.detect_duplicates(valid_transactions)
        duplicate_count = len(duplicates)
        if duplicate_count > 0:
            logger.info(f"Duplicate transactions detected: {duplicate_count}")
            audit_logger.log_duplicate_detection('transaction_level', duplicate_count)
        
        # Aplica IA para categorização e detecção de anomalias
        logger.info("Starting AI categorization and anomaly detection")
        categorized_transactions = ai_service.categorize_transactions_batch(valid_transactions)
        analyzed_transactions = ai_service.detect_anomalies(categorized_transactions)
        audit_logger.log_ai_operation('categorization_and_anomaly_detection', len(valid_transactions), True)
        
        # Salva as transações no banco de dados
        saved_count = 0
        total_entries_processed = len(analyzed_transactions)
        duplicate_entries_details = []
        
        for transaction_data in analyzed_transactions:
            # Verifica se a transação já existe (evita duplicatas) usando o serviço unificado
            account_id = ofx_data['account_info'].get('account_id', '')
            is_duplicate = DuplicateDetectionService.check_transaction_duplicate(
                account_id,
                transaction_data['date'],
                transaction_data['amount'],
                transaction_data['description']
            )
            
            if not is_duplicate:
                transaction = Transaction(
                    bank_name=ofx_data['bank_name'],
                    account_id=account_id,
                    transaction_id=transaction_data.get('transaction_id', ''),
                    date=transaction_data['date'],
                    amount=transaction_data['amount'],
                    description=transaction_data['description'],
                    transaction_type=transaction_data['transaction_type'],
                    balance=transaction_data.get('balance'),
                    category=transaction_data.get('category'),
                    is_anomaly=transaction_data.get('is_anomaly', False)
                )
                db.session.add(transaction)
                saved_count += 1
            else:
                # Adiciona detalhes da duplicata para o response
                duplicate_entries_details.append({
                    'date': transaction_data['date'],
                    'amount': transaction_data['amount'],
                    'description': transaction_data['description']
                })
        
        # Salva o histórico de upload
        upload_record = UploadHistory(
            filename=filename,
            bank_name=ofx_data['bank_name'],
            transactions_count=saved_count,
            status='success',
            file_hash=file_hash,
            duplicate_files_count=0,  # This file is not a duplicate
            duplicate_entries_count=duplicate_count,
            total_entries_processed=total_entries_processed
        )
        db.session.add(upload_record)
        
        db.session.commit()
        
        logger.info(f"OFX processing completed successfully. Saved {saved_count} transactions")
        audit_logger.log_file_processing_result(filename, True, saved_count, duplicate_count)
        audit_logger.log_database_operation('insert', 'transactions', saved_count, True)
        
        return jsonify({
            'success': True,
            'message': f'Arquivo processado com sucesso! {saved_count} transações importadas.',
            'data': {
                'bank_name': ofx_data['bank_name'],
                'account_info': ofx_data['account_info'],
                'items_imported': saved_count,
                'items_incomplete': 0,
                'duplicates_found': duplicate_count,
                'file_duplicate': False,
                'incomplete_items': [],
                'summary': ofx_data['summary'],
                'duplicate_details': {
                    'file_level': {
                        'is_duplicate': False,
                        'original_upload_date': None
                    },
                    'entry_level': {
                        'count': duplicate_count,
                        'details': duplicate_entries_details
                    }
                }
            }
        })
        
    finally:
        # Always clean up temporary files
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up temporary files: {str(cleanup_error)}")

@transactions_bp.route('/transactions', methods=['GET'])
@handle_errors
@rate_limit(max_requests=200, window_minutes=60)  # Higher limit for read operations
@validate_input_fields('bank', 'category', 'type', 'start_date', 'end_date')
def get_transactions():
    """
    Endpoint para listar transações com filtros
    """
    # Validate pagination parameters
    page = request.args.get('page', 1)
    per_page = request.args.get('per_page', 100)
    pagination_params = validate_pagination_params(page, per_page)
    
    # Calculate offset from page
    offset = (pagination_params['page'] - 1) * pagination_params['per_page']
    limit = pagination_params['per_page']
    
    # Parâmetros de filtro
    bank_name = request.args.get('bank')
    category = request.args.get('category')
    transaction_type = request.args.get('type')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Validate date parameters
    start_date_obj = None
    end_date_obj = None
    
    if start_date:
        try:
            start_date_obj = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if end_date:
        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    # Validate date range
    if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
        raise ValidationError("start_date must be before end_date")
    
    # Constrói a query
    query = Transaction.query
    
    if bank_name:
        query = query.filter(Transaction.bank_name == bank_name)
    
    if category:
        query = query.filter(Transaction.category == category)
    
    if transaction_type:
        if transaction_type not in ['credit', 'debit']:
            raise ValidationError("transaction_type must be 'credit' or 'debit'")
        query = query.filter(Transaction.transaction_type == transaction_type)
    
    if start_date_obj:
        query = query.filter(Transaction.date >= start_date_obj.date())
    
    if end_date_obj:
        query = query.filter(Transaction.date <= end_date_obj.date())
    
    # Ordena por data (mais recente primeiro)
    query = query.order_by(Transaction.date.desc())
    
    # Get total count before applying pagination
    total_count = query.count()
    
    # Aplica paginação
    transactions = query.offset(offset).limit(limit).all()
    
    logger.info(f"Retrieved {len(transactions)} transactions (total: {total_count})")
    
    return jsonify({
        'success': True,
        'data': {
            'transactions': [t.to_dict() for t in transactions],
            'total_count': total_count,
            'page': pagination_params['page'],
            'per_page': pagination_params['per_page'],
            'total_pages': (total_count + pagination_params['per_page'] - 1) // pagination_params['per_page']
        }
    })

@transactions_bp.route('/insights', methods=['GET'])
@handle_errors
def get_insights():
    """
    Endpoint para obter insights das transações
    """
    # Parâmetros de filtro (mesmo que get_transactions)
    bank_name = request.args.get('bank')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Validate date parameters
    start_date_obj = None
    end_date_obj = None
    
    if start_date:
        try:
            start_date_obj = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if end_date:
        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    # Validate date range
    if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
        raise ValidationError("start_date must be before end_date")
    
    # Constrói a query
    query = Transaction.query
    
    if bank_name:
        query = query.filter(Transaction.bank_name == bank_name)
    
    if start_date_obj:
        query = query.filter(Transaction.date >= start_date_obj.date())
    
    if end_date_obj:
        query = query.filter(Transaction.date <= end_date_obj.date())
    
    transactions = query.all()
    
    if not transactions:
        raise InsufficientDataError('insights generation', 1, 0)
    
    # Converte para formato de dicionário
    transactions_data = [t.to_dict() for t in transactions]
    
    # Gera insights
    ai_service = AIService()
    insights = ai_service.generate_insights(transactions_data)
    
    logger.info(f"Generated insights for {len(transactions)} transactions")
    
    return jsonify({
        'success': True,
        'data': insights
    })

@transactions_bp.route('/ai-insights', methods=['GET'])
@handle_errors
def get_ai_insights():
    """
    Endpoint para obter insights gerados por IA
    """
    # Parâmetros de filtro
    bank_name = request.args.get('bank')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Validate date parameters
    start_date_obj = None
    end_date_obj = None
    
    if start_date:
        try:
            start_date_obj = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if end_date:
        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    # Validate date range
    if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
        raise ValidationError("start_date must be before end_date")
    
    # Constrói a query (mesmo que get_insights)
    query = Transaction.query
    
    if bank_name:
        query = query.filter(Transaction.bank_name == bank_name)
    
    if start_date_obj:
        query = query.filter(Transaction.date >= start_date_obj.date())
    
    if end_date_obj:
        query = query.filter(Transaction.date <= end_date_obj.date())
    
    transactions = query.all()
    
    if not transactions:
        raise InsufficientDataError('AI insights generation', 1, 0)
    
    # Converte para formato de dicionário
    transactions_data = [t.to_dict() for t in transactions]
    
    # Gera insights com IA
    ai_service = AIService()
    ai_insights = ai_service.generate_ai_insights(transactions_data)
    
    logger.info(f"Generated AI insights for {len(transactions)} transactions")
    
    return jsonify({
        'success': True,
        'data': {
            'ai_insights': ai_insights
        }
    })

@transactions_bp.route('/summary', methods=['GET'])
def get_summary():
    """
    Endpoint para obter resumo geral das transações
    """
    try:
        # Estatísticas gerais
        total_transactions = Transaction.query.count()
        total_credits = db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.amount > 0).scalar() or 0
        total_debits = abs(db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.amount < 0).scalar() or 0)
        
        # Transações por banco
        banks = db.session.query(
            Transaction.bank_name,
            db.func.count(Transaction.id).label('count')
        ).group_by(Transaction.bank_name).all()
        
        # Transações por categoria
        categories = db.session.query(
            Transaction.category,
            db.func.count(Transaction.id).label('count'),
            db.func.sum(Transaction.amount).label('total')
        ).group_by(Transaction.category).all()
        
        # Anomalias
        anomalies_count = Transaction.query.filter(Transaction.is_anomaly == True).count()
        
        # Últimas transações
        recent_transactions = Transaction.query.order_by(Transaction.date.desc()).limit(5).all()
        
        return jsonify({
            'success': True,
            'data': {
                'overview': {
                    'total_transactions': total_transactions,
                    'total_credits': round(total_credits, 2),
                    'total_debits': round(total_debits, 2),
                    'net_flow': round(total_credits - total_debits, 2),
                    'anomalies_count': anomalies_count
                },
                'banks': [{'name': bank.bank_name, 'count': bank.count} for bank in banks],
                'categories': [
                    {
                        'name': cat.category,
                        'count': cat.count,
                        'total': round(cat.total or 0, 2)
                    } for cat in categories
                ],
                'recent_transactions': [t.to_dict() for t in recent_transactions]
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar resumo: {str(e)}'}), 500

@transactions_bp.route('/upload-history', methods=['GET'])
def get_upload_history():
    """
    Endpoint para obter histórico de uploads
    """
    try:
        uploads = UploadHistory.query.order_by(UploadHistory.upload_date.desc()).limit(20).all()
        
        return jsonify({
            'success': True,
            'data': {
                'uploads': [upload.to_dict() for upload in uploads]
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar histórico: {str(e)}'}), 500

@transactions_bp.route('/upload-xlsx', methods=['POST'])
@rate_limit(max_requests=50, window_minutes=60)  # Limit file uploads
@validate_file_upload(['xlsx'], max_size_mb=10)
def upload_xlsx():
    """
    Endpoint para upload e processamento de arquivos XLSX
    """
    try:
        logger.info("XLSX upload request received")
        
        if 'file' not in request.files:
            logger.warning("XLSX upload request without file")
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("XLSX upload request with empty filename")
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
        
        if not allowed_xlsx_file(file.filename):
            logger.warning(f"XLSX upload request with invalid file type: {file.filename}")
            return jsonify({'error': 'Tipo de arquivo não permitido. Use apenas .xlsx'}), 400
        
        # Salva o arquivo temporariamente
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)
        
        # Log file upload
        file_size = os.path.getsize(temp_path)
        logger.info(f"XLSX file uploaded: {filename} ({file_size} bytes)")
        
        # Verifica se o arquivo é duplicado
        file_hash = DuplicateDetectionService.calculate_file_hash(temp_path)
        audit_logger.log_file_upload(filename, 'XLSX', file_size=file_size, file_hash=file_hash)
        
        existing_file = DuplicateDetectionService.check_file_duplicate(file_hash)
        
        if existing_file:
            logger.warning(f"Duplicate XLSX file detected: {filename} (hash: {file_hash})")
            audit_logger.log_duplicate_detection('file_level', 1, {
                'filename': filename,
                'original_upload_date': existing_file.upload_date.isoformat() if existing_file.upload_date else None
            })
            
            # Remove o arquivo temporário
            os.remove(temp_path)
            os.rmdir(temp_dir)
            
            return jsonify({
                'success': False,
                'message': 'Arquivo já foi processado anteriormente.',
                'data': {
                    'items_imported': 0,
                    'items_incomplete': 0,
                    'duplicates_found': 0,
                    'file_duplicate': True,
                    'incomplete_items': [],
                    'original_upload_date': existing_file.upload_date.isoformat() if existing_file.upload_date else None,
                    'duplicate_details': {
                        'file_level': {
                            'is_duplicate': True,
                            'original_upload_date': existing_file.upload_date.isoformat() if existing_file.upload_date else None
                        },
                        'entry_level': {
                            'count': 0,
                            'details': []
                        }
                    }
                }
            })
        
        # Processa o arquivo XLSX
        logger.info(f"Starting XLSX processing for file: {filename}")
        processor = XLSXProcessor()
        financial_data = processor.parse_xlsx_file(temp_path)
        logger.info(f"XLSX parsed successfully. Entries found: {len(financial_data)}")
        
        # Salva os dados no banco de dados
        saved_count = 0
        incomplete_entries = []
        errors = []
        duplicate_entries_details = []
        duplicate_count = 0
        total_entries_processed = len(financial_data)
        
        for i, entry_data in enumerate(financial_data):
            try:
                # Verifica se a data é válida
                if entry_data['date'] is None:
                    # Adiciona à lista de entradas incompletas em vez de pular
                    entry_data['row_number'] = i + 1
                    entry_data['error'] = "Data inválida ou ausente"
                    incomplete_entries.append(entry_data)
                    continue
                
                # Verifica se a descrição está vazia
                if not entry_data['description'] or entry_data['description'].strip() == '':
                    # Adiciona à lista de entradas incompletas em vez de pular
                    entry_data['row_number'] = i + 1
                    entry_data['error'] = "Descrição ausente"
                    incomplete_entries.append(entry_data)
                    continue
                
                # Verifica se a entrada já existe (evita duplicatas)
                try:
                    is_duplicate = DuplicateDetectionService.check_financial_entry_duplicate(
                        entry_data['date'],
                        entry_data['amount'],
                        entry_data['description']
                    )
                    
                    if is_duplicate:
                        # Adiciona detalhes da duplicata para o response
                        duplicate_entries_details.append({
                            'date': entry_data['date'].isoformat() if entry_data['date'] else None,
                            'amount': entry_data['amount'],
                            'description': entry_data['description'],
                            'row_number': i + 1
                        })
                        duplicate_count += 1
                        continue
                        
                except Exception as duplicate_error:
                    # Se falhar a verificação de duplicata, trata como erro e não insere
                    error_msg = f"Erro na verificação de duplicata na linha {i + 1}: {str(duplicate_error)}"
                    errors.append(error_msg)
                    entry_data['row_number'] = i + 1
                    entry_data['error'] = f"Erro na verificação de duplicata: {str(duplicate_error)}"
                    incomplete_entries.append(entry_data)
                    continue
                
                # Se chegou até aqui, a entrada é válida e não é duplicata
                financial_entry = CompanyFinancial(
                    date=entry_data['date'],
                    description=entry_data['description'],
                    amount=entry_data['amount'],
                    category=entry_data['category'],
                    cost_center=entry_data['cost_center'],
                    department=entry_data['department'],
                    project=entry_data['project'],
                    transaction_type=entry_data['transaction_type']
                )
                db.session.add(financial_entry)
                saved_count += 1
                
            except Exception as e:
                # Adiciona erro específico para esta entrada
                error_msg = f"Erro na linha {i + 1}: {str(e)}"
                errors.append(error_msg)
                entry_data['row_number'] = i + 1
                entry_data['error'] = str(e)
                incomplete_entries.append(entry_data)
        
        # Salva o histórico de upload
        upload_record = UploadHistory(
            filename=filename,
            bank_name='xlsx_file',
            transactions_count=saved_count,
            status='success',
            file_hash=file_hash,
            duplicate_files_count=0,  # This file is not a duplicate
            duplicate_entries_count=duplicate_count,
            total_entries_processed=total_entries_processed
        )
        db.session.add(upload_record)
        
        db.session.commit()
        
        logger.info(f"XLSX processing completed. Saved: {saved_count}, Incomplete: {len(incomplete_entries)}, Duplicates: {duplicate_count}")
        audit_logger.log_file_processing_result(filename, True, saved_count, duplicate_count, errors)
        audit_logger.log_database_operation('insert', 'company_financial', saved_count, True)
        
        if duplicate_count > 0:
            audit_logger.log_duplicate_detection('entry_level', duplicate_count, {
                'details': duplicate_entries_details[:10]  # Log first 10 duplicates
            })
        
        # Remove o arquivo temporário
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        # Prepara a mensagem de retorno
        message = f'Arquivo processado com sucesso! {saved_count} entradas importadas.'
        if len(incomplete_entries) > 0:
            message += f' {len(incomplete_entries)} entradas incompletas encontradas e requerem correção.'
        if duplicate_count > 0:
            message += f' {duplicate_count} entradas duplicadas foram ignoradas.'
        
        response_data = {
            'success': True,
            'message': message,
            'data': {
                'items_imported': saved_count,
                'items_incomplete': len(incomplete_entries),
                'duplicates_found': duplicate_count,
                'file_duplicate': False,
                'incomplete_items': incomplete_entries,
                'duplicate_details': {
                    'file_level': {
                        'is_duplicate': False,
                        'original_upload_date': None
                    },
                    'entry_level': {
                        'count': duplicate_count,
                        'details': duplicate_entries_details
                    }
                }
            }
        }
        
        # Adiciona detalhes de erros se houver
        if errors:
            response_data['errors'] = errors
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing XLSX file {filename}: {str(e)}", exc_info=True)
        audit_logger.log_file_processing_result(filename, False, 0, 0, [str(e)])
        
        # Remove o arquivo temporário em caso de erro
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        
        return jsonify({'error': f'Erro ao processar arquivo: {str(e)}'}), 500

@transactions_bp.route('/upload-xlsx-corrected', methods=['POST'])
@rate_limit(max_requests=30, window_minutes=60)
@require_content_type('application/json')
@validate_financial_data('company_financial')
def upload_xlsx_corrected():
    """
    Endpoint para upload e processamento de dados corrigidos
    """
    try:
        # Obtém os dados corrigidos do corpo da requisição
        data = request.get_json()
        
        if not data or 'entries' not in data:
            return jsonify({'error': 'Nenhum dado enviado'}), 400
        
        corrected_entries = data['entries']
        
        # Salva os dados corrigidos no banco de dados
        saved_count = 0
        errors = []
        
        for i, entry_data in enumerate(corrected_entries):
            try:
                # Valida os dados corrigidos
                if not entry_data.get('date'):
                    errors.append(f"Entrada {i+1}: Data ausente")
                    continue
                
                if not entry_data.get('description') or entry_data['description'].strip() == '':
                    errors.append(f"Entrada {i+1}: Descrição ausente")
                    continue
                
                # Converte a data para o formato correto
                if isinstance(entry_data['date'], str):
                    from datetime import datetime
                    entry_data['date'] = datetime.fromisoformat(entry_data['date'].replace('Z', '+00:00')).date()
                
                financial_entry = CompanyFinancial(
                    date=entry_data['date'],
                    description=entry_data['description'],
                    amount=float(entry_data.get('amount', 0)),
                    category=entry_data.get('category', ''),
                    cost_center=entry_data.get('cost_center', ''),
                    department=entry_data.get('department', ''),
                    project=entry_data.get('project', ''),
                    transaction_type=entry_data.get('transaction_type', 'expense')
                )
                db.session.add(financial_entry)
                saved_count += 1
            except Exception as e:
                errors.append(f"Entrada {i+1}: Erro ao processar - {str(e)}")
        
        db.session.commit()
        
        # Prepara a mensagem de retorno
        message = f'Dados corrigidos processados com sucesso! {saved_count} entradas importadas.'
        if errors:
            message += f' {len(errors)} entradas com erros.'
        
        response_data = {
            'success': True,
            'message': message,
            'data': {
                'items_imported': saved_count,
                'items_incomplete': 0,
                'duplicates_found': 0,
                'file_duplicate': False,
                'incomplete_items': [],
                'duplicate_details': {
                    'file_level': {
                        'is_duplicate': False,
                        'original_upload_date': None
                    },
                    'entry_level': {
                        'count': 0,
                        'details': []
                    }
                }
            }
        }
        
        # Adiciona detalhes de erros se houver
        if errors:
            response_data['errors'] = errors
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Erro ao processar dados corrigidos: {str(e)}'}), 500

@transactions_bp.route('/company-financial', methods=['GET'])
def get_company_financial():
    """
    Endpoint para listar entradas financeiras da empresa com filtros
    """
    try:
        # Parâmetros de filtro
        transaction_type = request.args.get('type')  # expense, income, or all
        category = request.args.get('category')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Constrói a query
        query = CompanyFinancial.query
        
        if transaction_type and transaction_type != 'all':
            query = query.filter(CompanyFinancial.transaction_type == transaction_type)
        
        if category:
            query = query.filter(CompanyFinancial.category == category)
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        # Ordena por data (mais recente primeiro)
        query = query.order_by(CompanyFinancial.date.desc())
        
        # Aplica paginação
        entries = query.offset(offset).limit(limit).all()
        total_count = query.count()
        
        return jsonify({
            'success': True,
            'data': {
                'entries': [e.to_dict() for e in entries],
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar entradas financeiras: {str(e)}'}), 500

@transactions_bp.route('/company-financial/summary', methods=['GET'])
def get_company_financial_summary():
    """
    Endpoint para obter resumo geral das entradas financeiras
    """
    try:
        # Parâmetros de filtro
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Constrói a query base
        query = CompanyFinancial.query
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        # Aplica os filtros para estatísticas
        filtered_entries = query.all()
        
        # Estatísticas gerais
        total_entries = len(filtered_entries)
        total_income = sum(entry.amount for entry in filtered_entries if entry.transaction_type == 'income')
        total_expenses = sum(entry.amount for entry in filtered_entries if entry.transaction_type == 'expense')
        
        # Entradas por categoria
        category_stats = {}
        for entry in filtered_entries:
            if entry.category not in category_stats:
                category_stats[entry.category] = {'count': 0, 'total': 0}
            category_stats[entry.category]['count'] += 1
            category_stats[entry.category]['total'] += entry.amount
        
        # Últimas entradas
        recent_entries = sorted(filtered_entries, key=lambda x: x.date, reverse=True)[:5]
        
        return jsonify({
            'success': True,
            'data': {
                'overview': {
                    'total_entries': total_entries,
                    'total_income': round(total_income, 2),
                    'total_expenses': round(total_expenses, 2),
                    'net_flow': round(total_income - total_expenses, 2)
                },
                'categories': [
                    {
                        'name': category,
                        'count': stats['count'],
                        'total': round(stats['total'], 2)
                    } for category, stats in category_stats.items()
                ],
                'recent_entries': [e.to_dict() for e in recent_entries]
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar resumo: {str(e)}'}), 500

@transactions_bp.route('/ai/train', methods=['POST'])
def train_ai_model():
    """
    Endpoint para treinar o modelo de IA com dados financeiros da empresa
    """
    try:
        # Parâmetros de filtro para selecionar dados de treinamento
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        category = request.args.get('category')
        
        # Constrói a query para dados de treinamento
        query = CompanyFinancial.query
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        if category:
            query = query.filter(CompanyFinancial.category == category)
        
        # Obtém os dados de treinamento
        training_data = query.all()
        
        if not training_data:
            return jsonify({
                'success': False,
                'error': 'Nenhum dado encontrado para treinamento'
            }), 400
        
        # Converte para formato de dicionário
        training_data_dicts = [entry.to_dict() for entry in training_data]
        
        # Treina o modelo
        ai_service = AIService()
        result = ai_service.train_custom_model(training_data_dicts)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Modelo treinado com sucesso',
                'accuracy': result['accuracy']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        return jsonify({'error': f'Erro ao treinar modelo: {str(e)}'}), 500

@transactions_bp.route('/ai/categorize-financial', methods=['POST'])
def categorize_financial_with_ai():
    """
    Endpoint para categorizar entradas financeiras usando o modelo treinado
    """
    try:
        # Obtém os dados para categorização
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({'error': 'Descrição não fornecida'}), 400
        
        description = data['description']
        
        # Categoriza usando o modelo treinado
        ai_service = AIService()
        category = ai_service.categorize_with_custom_model(description)
        
        return jsonify({
            'success': True,
            'data': {
                'category': category
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao categorizar: {str(e)}'}), 500

@transactions_bp.route('/ai/predictions', methods=['GET'])
def get_financial_predictions():
    """
    Endpoint para obter previsões financeiras baseadas em dados históricos
    """
    try:
        # Parâmetros para dados históricos
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        periods = request.args.get('periods', 12, type=int)
        
        # Constrói a query para dados históricos
        query = CompanyFinancial.query
        
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
            query = query.filter(CompanyFinancial.date >= start_date_obj.date())
        
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
            query = query.filter(CompanyFinancial.date <= end_date_obj.date())
        
        # Obtém os dados históricos
        historical_data = query.all()
        
        if not historical_data:
            return jsonify({
                'success': True,
                'data': {
                    'message': 'Nenhum dado encontrado para predição'
                }
            })
        
        # Converte para formato de dicionário
        historical_data_dicts = [entry.to_dict() for entry in historical_data]
        
        # Gera previsões
        ai_service = AIService()
        predictions = ai_service.predict_financial_trends(historical_data_dicts, periods)
        
        if predictions['success']:
            return jsonify({
                'success': True,
                'data': predictions['data']
            })
        else:
            return jsonify({
                'success': False,
                'error': predictions['error']
            }), 500
            
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar previsões: {str(e)}'}), 500

@transactions_bp.route('/reconciliation', methods=['POST'])
@handle_errors
@with_resource_check
@rate_limit(max_requests=20, window_minutes=60)  # Lower limit for intensive operations
@validate_input_fields('start_date', 'end_date')
def start_reconciliation():
    """
    Endpoint para iniciar o processo de reconciliação entre transações bancárias e entradas financeiras
    """
    # Parâmetros para filtrar dados
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Validate date parameters
    start_date_obj = None
    end_date_obj = None
    
    if start_date:
        try:
            start_date_obj = datetime.fromisoformat(start_date)
        except ValueError:
            raise ValidationError("Invalid start_date format. Use ISO format (YYYY-MM-DD)")
    
    if end_date:
        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except ValueError:
            raise ValidationError("Invalid end_date format. Use ISO format (YYYY-MM-DD)")
    
    # Validate date range
    if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
        raise ValidationError("start_date must be before end_date")
    
    # Constrói queries para dados bancários e financeiros
    bank_query = Transaction.query
    company_query = CompanyFinancial.query
    
    if start_date_obj:
        bank_query = bank_query.filter(Transaction.date >= start_date_obj.date())
        company_query = company_query.filter(CompanyFinancial.date >= start_date_obj.date())
    
    if end_date_obj:
        bank_query = bank_query.filter(Transaction.date <= end_date_obj.date())
        company_query = company_query.filter(CompanyFinancial.date <= end_date_obj.date())
    
    # Obtém os dados
    bank_transactions = bank_query.all()
    company_entries = company_query.all()
    
    if not bank_transactions:
        raise InsufficientDataError('reconciliation', 1, 0)
    
    if not company_entries:
        raise InsufficientDataError('reconciliation', 1, 0)
    
    # Realiza a reconciliação
    logger.info(f"Starting reconciliation with {len(bank_transactions)} bank transactions and {len(company_entries)} company entries")
    reconciliation_service = ReconciliationService()
    matches = reconciliation_service.find_matches(bank_transactions, company_entries)
    records = reconciliation_service.create_reconciliation_records(matches)
    
    logger.info(f"Reconciliation completed. {len(records)} matches found")
    audit_logger.log_reconciliation_operation('start', len(records), 'completed', {
        'bank_transactions': len(bank_transactions),
        'company_entries': len(company_entries),
        'date_range': {'start': start_date, 'end': end_date}
    })
    
    return jsonify({
        'success': True,
        'message': f'Reconciliação iniciada com sucesso. {len(records)} correspondências encontradas.',
        'data': {
            'matches_count': len(records),
            'bank_transactions_count': len(bank_transactions),
            'company_entries_count': len(company_entries)
        }
    })

@transactions_bp.route('/reconciliation/pending', methods=['GET'])
def get_pending_reconciliations():
    """
    Endpoint para obter registros de reconciliação pendentes
    """
    try:
        reconciliation_service = ReconciliationService()
        pending_records = reconciliation_service.get_pending_reconciliations()
        
        return jsonify({
            'success': True,
            'data': {
                'records': [record.to_dict() for record in pending_records]
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao buscar reconciliações pendentes: {str(e)}'}), 500

@transactions_bp.route('/reconciliation/<int:reconciliation_id>/confirm', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@sanitize_path_params('reconciliation_id')
def confirm_reconciliation(reconciliation_id):
    """
    Endpoint para confirmar uma reconciliação
    """
    # Validate reconciliation_id
    if reconciliation_id <= 0:
        raise ValidationError("Invalid reconciliation ID")
    
    reconciliation_service = ReconciliationService()
    success = reconciliation_service.confirm_reconciliation(reconciliation_id)
    
    logger.info(f"Reconciliation {reconciliation_id} confirmed successfully")
    
    return jsonify({
        'success': True,
        'message': 'Reconciliação confirmada com sucesso',
        'data': {
            'reconciliation_id': reconciliation_id
        }
    })

@transactions_bp.route('/reconciliation/<int:reconciliation_id>/reject', methods=['POST'])
@handle_errors
@rate_limit(max_requests=100, window_minutes=60)
@sanitize_path_params('reconciliation_id')
def reject_reconciliation(reconciliation_id):
    """
    Endpoint para rejeitar uma reconciliação
    """
    # Validate reconciliation_id
    if reconciliation_id <= 0:
        raise ValidationError("Invalid reconciliation ID")
    
    reconciliation_service = ReconciliationService()
    success = reconciliation_service.reject_reconciliation(reconciliation_id)
    
    logger.info(f"Reconciliation {reconciliation_id} rejected successfully")
    
    return jsonify({
        'success': True,
        'message': 'Reconciliação rejeitada com sucesso',
        'data': {
            'reconciliation_id': reconciliation_id
        }
    })

@transactions_bp.route('/reconciliation/report', methods=['GET'])
def get_reconciliation_report():
    """
    Endpoint para obter relatório de reconciliação
    """
    try:
        reconciliation_service = ReconciliationService()
        report = reconciliation_service.get_reconciliation_report()
        
        return jsonify({
            'success': True,
            'data': report
        })
        
    except Exception as e:
        return jsonify({'error': f'Erro ao gerar relatório: {str(e)}'}), 500

