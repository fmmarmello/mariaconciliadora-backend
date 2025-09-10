"""
Real-time Processing Service

Handles real-time processing of reconciliation tasks including:
- Background job processing
- Real-time notifications
- Event-driven workflows
- Performance monitoring
- Queue management
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
from sqlalchemy import and_, or_
from src.models.transaction import db, ReconciliationRecord, Transaction, UploadHistory
from src.models.user import User
from src.services.escalation_service import EscalationService
from src.services.audit_trail_service import AuditTrailService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class ProcessingPriority(Enum):
    """Processing priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class JobStatus(Enum):
    """Background job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class EventType(Enum):
    """Real-time event types"""
    TRANSACTION_UPLOADED = "transaction_uploaded"
    RECONCILIATION_MATCHED = "reconciliation_matched"
    RECONCILIATION_APPROVED = "reconciliation_approved"
    RECONCILIATION_REJECTED = "reconciliation_rejected"
    ESCALATION_TRIGGERED = "escalation_triggered"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_METRIC = "performance_metric"

@dataclass
class ProcessingJob:
    """Background processing job"""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # 5 minutes default

@dataclass
class RealtimeEvent:
    """Real-time event for notifications"""
    id: str
    event_type: EventType
    data: Dict[str, Any]
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    expires_at: Optional[datetime] = None

class RealtimeProcessingService:
    """Service for real-time processing and background jobs"""
    
    def __init__(self, max_workers: int = 10, redis_url: Optional[str] = None):
        self.max_workers = max_workers
        self.redis_url = redis_url
        
        # Thread pools for different priority levels
        self.executors = {
            ProcessingPriority.CRITICAL: ThreadPoolExecutor(max_workers=2),
            ProcessingPriority.URGENT: ThreadPoolExecutor(max_workers=2),
            ProcessingPriority.HIGH: ThreadPoolExecutor(max_workers=3),
            ProcessingPriority.NORMAL: ThreadPoolExecutor(max_workers=2),
            ProcessingPriority.LOW: ThreadPoolExecutor(max_workers=1),
        }
        
        # Job queues
        self.job_queues = {
            ProcessingPriority.CRITICAL: Queue(),
            ProcessingPriority.URGENT: Queue(),
            ProcessingPriority.HIGH: Queue(),
            ProcessingPriority.NORMAL: Queue(),
            ProcessingPriority.LOW: Queue(),
        }
        
        # Active jobs tracking
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.job_futures: Dict[str, Future] = {}
        
        # Event system
        self.event_queue = Queue()
        self.event_subscribers: Dict[EventType, List[Callable]] = {}
        
        # Redis for pub/sub (if available)
        self.redis_client = None
        self.redis_pubsub = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_pubsub = self.redis_client.pubsub()
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {str(e)}")
                self.redis_client = None
        
        # Performance metrics
        self.metrics = {
            'jobs_processed': 0,
            'jobs_failed': 0,
            'events_processed': 0,
            'avg_processing_time': 0,
            'queue_sizes': {}
        }
        
        # Initialize services
        self.escalation_service = EscalationService()
        self.audit_service = AuditTrailService()
        
        # Start background threads
        self._running = True
        self._start_background_threads()
        
        logger.info("Real-time processing service initialized")
    
    def _start_background_threads(self):
        """Start background processing threads"""
        # Job processor threads
        for priority, executor in self.executors.items():
            thread = threading.Thread(
                target=self._job_processor,
                args=(priority, executor),
                daemon=True
            )
            thread.start()
            logger.info(f"Started job processor for {priority.value} priority")
        
        # Event processor thread
        event_thread = threading.Thread(
            target=self._event_processor,
            daemon=True
        )
        event_thread.start()
        logger.info("Started event processor")
        
        # Metrics collector thread
        metrics_thread = threading.Thread(
            target=self._metrics_collector,
            daemon=True
        )
        metrics_thread.start()
        logger.info("Started metrics collector")
        
        # Cleanup thread
        cleanup_thread = threading.Thread(
            target=self._cleanup_old_jobs,
            daemon=True
        )
        cleanup_thread.start()
        logger.info("Started cleanup thread")
    
    def _job_processor(self, priority: ProcessingPriority, executor: ThreadPoolExecutor):
        """Process jobs from the queue"""
        while self._running:
            try:
                job = self.job_queues[priority].get(timeout=1)
                if job.status == JobStatus.CANCELLED:
                    continue
                
                # Update job status
                job.status = JobStatus.RUNNING
                job.started_at = datetime.utcnow()
                
                # Execute job with timeout
                future = executor.submit(
                    self._execute_job,
                    job
                )
                self.job_futures[job.id] = future
                
                # Wait for completion or timeout
                try:
                    future.result(timeout=job.timeout)
                except Exception as e:
                    self._handle_job_failure(job, str(e))
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in job processor: {str(e)}")
    
    def _execute_job(self, job: ProcessingJob):
        """Execute a single job"""
        try:
            logger.info(f"Executing job: {job.name} ({job.id})")
            
            # Execute the job function
            result = job.func(*job.args, **job.kwargs)
            
            # Update job status
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result
            
            # Update metrics
            self.metrics['jobs_processed'] += 1
            
            # Log completion
            logger.info(f"Job completed: {job.name} ({job.id})")
            
            # Emit completion event
            self.emit_event(
                EventType.SYSTEM_ALERT,
                {
                    'job_id': job.id,
                    'job_name': job.name,
                    'status': 'completed',
                    'processing_time': (job.completed_at - job.started_at).total_seconds()
                }
            )
            
        except Exception as e:
            self._handle_job_failure(job, str(e))
    
    def _handle_job_failure(self, job: ProcessingJob, error: str):
        """Handle job failure with retry logic"""
        job.error = error
        job.retry_count += 1
        
        if job.retry_count < job.max_retries:
            # Retry the job
            job.status = JobStatus.RETRYING
            self.job_queues[job.priority].put(job)
            logger.warning(f"Retrying job {job.name} ({job.id}) - attempt {job.retry_count}")
        else:
            # Mark as failed
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            
            # Update metrics
            self.metrics['jobs_failed'] += 1
            
            # Log failure
            logger.error(f"Job failed: {job.name} ({job.id}) - {error}")
            
            # Emit failure event
            self.emit_event(
                EventType.SYSTEM_ALERT,
                {
                    'job_id': job.id,
                    'job_name': job.name,
                    'status': 'failed',
                    'error': error
                }
            )
    
    def _event_processor(self):
        """Process real-time events"""
        while self._running:
            try:
                event = self.event_queue.get(timeout=1)
                
                # Process event
                self._process_event(event)
                
                # Update metrics
                self.metrics['events_processed'] += 1
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in event processor: {str(e)}")
    
    def _process_event(self, event: RealtimeEvent):
        """Process a single event"""
        try:
            # Log event
            logger.info(f"Processing event: {event.event_type.value}")
            
            # Notify subscribers
            if event.event_type in self.event_subscribers:
                for callback in self.event_subscribers[event.event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in event callback: {str(e)}")
            
            # Publish to Redis if available
            if self.redis_client:
                self._publish_to_redis(event)
            
            # Handle specific event types
            if event.event_type == EventType.TRANSACTION_UPLOADED:
                self._handle_transaction_upload(event)
            elif event.event_type == EventType.RECONCILIATION_MATCHED:
                self._handle_reconciliation_match(event)
            elif event.event_type == EventType.ESCALATION_TRIGGERED:
                self._handle_escalation(event)
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_type.value}: {str(e)}")
    
    def _handle_transaction_upload(self, event: RealtimeEvent):
        """Handle transaction upload event"""
        try:
            # Start background reconciliation process
            upload_id = event.data.get('upload_id')
            if upload_id:
                self.submit_job(
                    name=f"Reconcile upload {upload_id}",
                    func=self._reconcile_upload,
                    args=(upload_id,),
                    priority=ProcessingPriority.HIGH
                )
        except Exception as e:
            logger.error(f"Error handling transaction upload: {str(e)}")
    
    def _handle_reconciliation_match(self, event: RealtimeEvent):
        """Handle reconciliation match event"""
        try:
            # Check for escalation triggers
            reconciliation_id = event.data.get('reconciliation_id')
            if reconciliation_id:
                self.submit_job(
                    name=f"Check escalation for {reconciliation_id}",
                    func=self._check_escalation_triggers,
                    args=(reconciliation_id,),
                    priority=ProcessingPriority.NORMAL
                )
        except Exception as e:
            logger.error(f"Error handling reconciliation match: {str(e)}")
    
    def _handle_escalation(self, event: RealtimeEvent):
        """Handle escalation event"""
        try:
            # Send notifications
            escalation_data = event.data
            self.emit_event(
                EventType.SYSTEM_ALERT,
                {
                    'type': 'escalation',
                    'severity': escalation_data.get('severity', 'medium'),
                    'message': escalation_data.get('reason', 'Escalation triggered'),
                    'target_user': escalation_data.get('target_user_name')
                }
            )
        except Exception as e:
            logger.error(f"Error handling escalation: {str(e)}")
    
    def _publish_to_redis(self, event: RealtimeEvent):
        """Publish event to Redis pub/sub"""
        try:
            if self.redis_client:
                channel = f"events:{event.event_type.value}"
                message = {
                    'id': event.id,
                    'type': event.event_type.value,
                    'data': event.data,
                    'timestamp': event.timestamp.isoformat(),
                    'priority': event.priority.value
                }
                self.redis_client.publish(channel, json.dumps(message))
        except Exception as e:
            logger.error(f"Error publishing to Redis: {str(e)}")
    
    def _metrics_collector(self):
        """Collect and report metrics"""
        while self._running:
            try:
                time.sleep(60)  # Collect metrics every minute
                
                # Calculate queue sizes
                for priority, queue in self.job_queues.items():
                    self.metrics['queue_sizes'][priority.value] = queue.qsize()
                
                # Log metrics
                logger.info(f"Metrics: {self.metrics}")
                
                # Emit metrics event
                self.emit_event(
                    EventType.PERFORMANCE_METRIC,
                    self.metrics.copy()
                )
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {str(e)}")
    
    def _cleanup_old_jobs(self):
        """Clean up old completed jobs"""
        while self._running:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                jobs_to_remove = []
                
                for job_id, job in self.active_jobs.items():
                    if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and
                        job.completed_at and job.completed_at < cutoff_time):
                        jobs_to_remove.append(job_id)
                
                # Remove old jobs
                for job_id in jobs_to_remove:
                    del self.active_jobs[job_id]
                    if job_id in self.job_futures:
                        del self.job_futures[job_id]
                
                if jobs_to_remove:
                    logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
                
            except Exception as e:
                logger.error(f"Error in cleanup thread: {str(e)}")
    
    def submit_job(self, name: str, func: Callable, args: tuple = (), 
                   kwargs: dict = None, priority: ProcessingPriority = ProcessingPriority.NORMAL,
                   timeout: int = 300) -> str:
        """Submit a background job for processing"""
        if kwargs is None:
            kwargs = {}
        
        job_id = f"job_{int(time.time() * 1000)}_{hash(name) % 10000}"
        
        job = ProcessingJob(
            id=job_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        self.active_jobs[job_id] = job
        self.job_queues[priority].put(job)
        
        logger.info(f"Job submitted: {name} ({job_id}) with priority {priority.value}")
        return job_id
    
    def emit_event(self, event_type: EventType, data: Dict[str, Any], 
                   priority: ProcessingPriority = ProcessingPriority.NORMAL,
                   user_id: Optional[int] = None, session_id: Optional[str] = None):
        """Emit a real-time event"""
        event_id = f"event_{int(time.time() * 1000)}_{hash(event_type.value) % 10000}"
        
        event = RealtimeEvent(
            id=event_id,
            event_type=event_type,
            data=data,
            priority=priority,
            user_id=user_id,
            session_id=session_id
        )
        
        self.event_queue.put(event)
        
        logger.debug(f"Event emitted: {event_type.value} ({event_id})")
        return event_id
    
    def subscribe_to_event(self, event_type: EventType, callback: Callable):
        """Subscribe to real-time events"""
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []
        
        self.event_subscribers[event_type].append(callback)
        logger.info(f"Subscribed to {event_type.value} events")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        job = self.active_jobs.get(job_id)
        if job:
            return {
                'id': job.id,
                'name': job.name,
                'status': job.status.value,
                'priority': job.priority.value,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'retry_count': job.retry_count,
                'error': job.error
            }
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        job = self.active_jobs.get(job_id)
        if job and job.status == JobStatus.RUNNING:
            job.status = JobStatus.CANCELLED
            
            # Cancel future if possible
            if job_id in self.job_futures:
                future = self.job_futures[job_id]
                future.cancel()
            
            logger.info(f"Job cancelled: {job.name} ({job_id})")
            return True
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
    
    def shutdown(self):
        """Shutdown the processing service"""
        logger.info("Shutting down real-time processing service")
        self._running = False
        
        # Shutdown executors
        for executor in self.executors.values():
            executor.shutdown(wait=True)
        
        logger.info("Real-time processing service shutdown complete")
    
    # Background job methods
    def _reconcile_upload(self, upload_id: int):
        """Background job to reconcile uploaded transactions"""
        try:
            # Get upload history
            upload = UploadHistory.query.get(upload_id)
            if not upload:
                logger.error(f"Upload not found: {upload_id}")
                return
            
            # Get transactions for this upload
            transactions = Transaction.query.filter_by(
                bank_name=upload.bank_name,
                created_at=upload.upload_date
            ).all()
            
            # Process each transaction
            for transaction in transactions:
                # This would integrate with the reconciliation service
                logger.info(f"Processing transaction {transaction.id}")
                
                # Emit progress event
                self.emit_event(
                    EventType.SYSTEM_ALERT,
                    {
                        'type': 'progress',
                        'upload_id': upload_id,
                        'transaction_id': transaction.id,
                        'progress': 'processing'
                    }
                )
            
            logger.info(f"Reconciliation completed for upload {upload_id}")
            
        except Exception as e:
            logger.error(f"Error reconciling upload {upload_id}: {str(e)}")
            raise
    
    def _check_escalation_triggers(self, reconciliation_id: int):
        """Background job to check escalation triggers"""
        try:
            # Check for escalations
            escalations = self.escalation_service.check_escalations()
            
            if escalations:
                for escalation in escalations:
                    self.emit_event(
                        EventType.ESCALATION_TRIGGERED,
                        {
                            'reconciliation_id': reconciliation_id,
                            'escalation_id': escalation.id,
                            'severity': escalation.severity,
                            'reason': escalation.reason,
                            'target_user_name': escalation.target_user_name
                        }
                    )
            
        except Exception as e:
            logger.error(f"Error checking escalation triggers: {str(e)}")
            raise
    
    # Helper methods
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        status = {}
        for priority, queue in self.job_queues.items():
            status[priority.value] = {
                'size': queue.qsize(),
                'active_workers': len([f for f in self.job_futures.values() if not f.done()])
            }
        return status
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of active jobs"""
        return [self.get_job_status(job_id) for job_id in self.active_jobs.keys()]