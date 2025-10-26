from flask import Blueprint, jsonify, request
import os
import logging
from src.services.test_data_deletion_service import test_data_deletion_service

# Configure logging
logger = logging.getLogger(__name__)

test_data_bp = Blueprint('test_data', __name__)

# Check if test data deletion is enabled
ENABLE_TEST_DATA_DELETION = os.getenv('ENABLE_TEST_DATA_DELETION', 'false').lower() == 'true'

def require_test_data_deletion_enabled():
    """Check if test data deletion feature is enabled"""
    if not ENABLE_TEST_DATA_DELETION:
        return False, "Test data deletion feature is disabled"
    return True, ""

def require_admin_access():
    """
    Check if user has admin access
    For now, we'll allow access in development mode or when specifically enabled
    In a production environment, this would check actual user permissions
    """
    # In development mode, allow access
    if os.getenv('FLASK_ENV') == 'development':
        return True, ""
    
    # If we had user authentication, we would check for admin role here
    # For now, we'll assume that if the feature is enabled, the user has appropriate access
    return True, ""

@test_data_bp.route('/test-data', methods=['DELETE'])
def delete_test_data():
    """
    DELETE endpoint for test data deletion with triple check mechanism
    
    Query Parameters:
    - mode: preview|confirmation|execution (required)
    - days_old: int (optional, default: 30)
    - force: boolean (optional, default: false)
    
    Returns:
    - 200: Success with details
    - 400: Bad request (missing parameters, invalid mode)
    - 403: Forbidden (feature disabled or insufficient permissions)
    - 500: Internal server error
    """
    try:
        # Check if test data deletion is enabled
        enabled, message = require_test_data_deletion_enabled()
        if not enabled:
            logger.warning(f"Test data deletion access attempt when disabled: {message}")
            return jsonify({
                'success': False,
                'error': message
            }), 403
        
        # Check if user has appropriate access
        has_access, message = require_admin_access()
        if not has_access:
            logger.warning(f"Unauthorized test data deletion attempt: {message}")
            return jsonify({
                'success': False,
                'error': 'Insufficient permissions for test data deletion'
            }), 403
        
        # Get data from request body
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Missing request body'}), 400

        mode = data.get('mode')
        days_old = data.get('days_old', 30)
        force = str(data.get('force', 'false')).lower() == 'true'
        include_recent = str(data.get('include_recent', 'false')).lower() == 'true'
        clear_users = str(data.get('clear_users', include_recent)).lower() == 'true'
        preserve_user_ids = data.get('preserve_user_ids', [1])
        preserve_org_ids = data.get('preserve_org_ids', [1])
        
        def _normalize_ids(raw):
            if raw is None:
                return []
            if isinstance(raw, (int, float)):
                return [int(raw)]
            if isinstance(raw, str):
                if not raw:
                    return []
                return [int(x) for x in raw.split(',')]
            if isinstance(raw, list):
                return [int(x) for x in raw if x is not None]
            return []
        
        preserve_user_ids = _normalize_ids(preserve_user_ids) or [1]
        preserve_org_ids = _normalize_ids(preserve_org_ids)
        if preserve_org_ids == []:
            preserve_org_ids = [1]
        
        # Validate mode parameter
        if not mode or mode not in ['preview', 'confirmation', 'execution']:
            logger.warning("Invalid or missing mode parameter for test data deletion")
            return jsonify({
                'success': False,
                'error': 'Invalid mode parameter. Must be one of: preview, confirmation, execution'
            }), 400
        
        logger.info(f"Test data deletion requested with mode: {mode}, days_old: {days_old}, force: {force}")
        
        # Handle the triple check mechanism based on mode
        if mode == 'preview':
            # First call: Return a list of data that would be deleted (preview)
            test_data_summary = test_data_deletion_service.identify_test_data(
                days_old,
                include_recent=include_recent,
                preserve_user_ids=preserve_user_ids,
                preserve_org_ids=preserve_org_ids,
                clear_users=clear_users
            )
            total_count = test_data_summary.get('total_count', 0)
            
            if total_count == 0:
                logger.info("No test data found for deletion in preview mode")
                return jsonify({
                    'success': True,
                    'message': 'No test data found for deletion',
                    'data': test_data_summary
                })
            
            logger.info(f"Preview mode: Found {total_count} test data records")
            return jsonify({
                'success': True,
                'message': f'Found {total_count} test data records that would be deleted',
                'data': test_data_summary
            })
            
        elif mode == 'confirmation':
            # Second call: Confirm the deletion with a warning (confirmation)
            test_data_details = test_data_deletion_service.get_test_data_details(
                days_old,
                include_recent=include_recent,
                preserve_user_ids=preserve_user_ids,
                preserve_org_ids=preserve_org_ids,
                clear_users=clear_users
            )
            total_count = test_data_details.get('total_count', 0)
            
            if total_count == 0:
                logger.info("No test data found for deletion in confirmation mode")
                return jsonify({
                    'success': True,
                    'message': 'No test data found for deletion',
                    'data': test_data_details
                })
            
            logger.info(f"Confirmation mode: Detailed view of {total_count} test data records")
            return jsonify({
                'success': True,
                'message': 'Please review the detailed test data before proceeding with deletion',
                'warning': 'This action will permanently delete the data listed below. This operation cannot be undone.',
                'data': test_data_details
            })
            
        elif mode == 'execution':
            # Third call: Execute the actual deletion (execution)
            if not force:
                logger.info("Execution mode requested without force flag")
                return jsonify({
                    'success': False,
                    'error': 'Force parameter required for execution mode',
                    'message': 'To execute deletion, you must set force=true parameter'
                }), 400
            
            logger.info(f"Execution mode: Starting deletion of test data older than {days_old} days")
            
            # Perform the actual deletion
            result = test_data_deletion_service._delete_test_data(
                days_old,
                include_recent=include_recent,
                preserve_user_ids=preserve_user_ids,
                preserve_org_ids=preserve_org_ids,
                clear_users=clear_users
            )
            
            if result:
                logger.info("Test data deletion completed successfully")
                return jsonify({
                    'success': True,
                    'message': 'Test data deletion completed successfully'
                })
            else:
                logger.error("Test data deletion failed")
                return jsonify({
                    'success': False,
                    'error': 'Test data deletion failed'
                }), 500
                
    except Exception as e:
        logger.error(f"Error during test data deletion: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error during test data deletion: {str(e)}'
        }), 500

# Lightweight status check endpoint to inform the frontend if test data deletion is enabled
@test_data_bp.route('/test-data/enabled', methods=['GET'])
def test_data_deletion_enabled():
    try:
        enabled, _ = require_test_data_deletion_enabled()
        has_access, _ = require_admin_access()
        return jsonify({
            'success': True,
            'enabled': bool(enabled and has_access)
        })
    except Exception as e:
        logger.error(f"Error checking test data deletion enabled: {str(e)}")
        return jsonify({
            'success': True,
            'enabled': False
        })
