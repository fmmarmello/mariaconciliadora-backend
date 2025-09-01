import time
import statistics
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PreprocessingMetrics:
    """
    Comprehensive metrics and monitoring system for text preprocessing
    """

    def __init__(self, max_history_size: int = 1000):
        """
        Initialize preprocessing metrics

        Args:
            max_history_size: Maximum number of historical records to keep
        """
        self.max_history_size = max_history_size

        # Core metrics
        self.total_processed = 0
        self.total_errors = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0

        # Performance metrics
        self.processing_times = deque(maxlen=max_history_size)
        self.quality_scores = deque(maxlen=max_history_size)
        self.text_lengths = deque(maxlen=max_history_size)

        # Step-level metrics
        self.step_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'errors': 0,
            'avg_time': 0.0,
            'success_rate': 0.0
        })

        # Quality metrics
        self.quality_metrics = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'avg': 0.0,
            'min': float('inf'),
            'max': 0.0,
            'distribution': defaultdict(int)
        })

        # Error tracking
        self.error_types = defaultdict(int)
        self.error_history = deque(maxlen=max_history_size)

        # Text characteristics
        self.language_distribution = defaultdict(int)
        self.domain_distribution = defaultdict(int)
        self.entity_counts = defaultdict(int)

        # Time-based metrics
        self.hourly_stats = defaultdict(lambda: {
            'processed': 0,
            'errors': 0,
            'avg_time': 0.0,
            'avg_quality': 0.0
        })

        # Start time for uptime calculation
        self.start_time = time.time()

        logger.info("PreprocessingMetrics initialized")

    def record_processing(self, result: Dict[str, Any], processing_time: float):
        """
        Record a preprocessing operation

        Args:
            result: Preprocessing result dictionary
            processing_time: Time taken for processing
        """
        try:
            self.total_processed += 1

            # Record basic metrics
            self.processing_times.append(processing_time)
            self.text_lengths.append(len(result.get('original_text', '')))

            # Record quality score
            quality_score = result.get('quality_score', 0.0)
            self.quality_scores.append(quality_score)

            # Update quality metrics
            self._update_quality_metrics(quality_score)

            # Record step metrics
            intermediate_results = result.get('intermediate_results', {})
            for step_name, step_result in intermediate_results.items():
                if isinstance(step_result, dict):
                    step_time = step_result.get('processing_time', 0.0)
                    step_success = step_result.get('success', True)

                    self.step_metrics[step_name]['count'] += 1
                    self.step_metrics[step_name]['total_time'] += step_time

                    if not step_success:
                        self.step_metrics[step_name]['errors'] += 1

                    # Update averages
                    count = self.step_metrics[step_name]['count']
                    self.step_metrics[step_name]['avg_time'] = (
                        self.step_metrics[step_name]['total_time'] / count
                    )
                    self.step_metrics[step_name]['success_rate'] = (
                        (count - self.step_metrics[step_name]['errors']) / count
                    )

            # Record errors
            if not result.get('success', True):
                self.total_errors += 1
                error_msg = result.get('error', 'Unknown error')
                self.error_types[error_msg] += 1
                self.error_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'error': error_msg,
                    'text_length': len(result.get('original_text', ''))
                })

            # Record text characteristics
            self._record_text_characteristics(result)

            # Record hourly stats
            self._record_hourly_stats(result, processing_time)

        except Exception as e:
            logger.warning(f"Error recording processing metrics: {str(e)}")

    def record_cache_hit(self):
        """Record a cache hit"""
        self.total_cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss"""
        self.total_cache_misses += 1

    def _update_quality_metrics(self, quality_score: float):
        """Update quality score metrics"""
        qm = self.quality_metrics['overall_quality']
        qm['count'] += 1
        qm['sum'] += quality_score
        qm['avg'] = qm['sum'] / qm['count']
        qm['min'] = min(qm['min'], quality_score)
        qm['max'] = max(qm['max'], quality_score)

        # Update distribution (bucket by 0.1 intervals)
        bucket = int(quality_score * 10) / 10
        qm['distribution'][bucket] += 1

    def _record_text_characteristics(self, result: Dict[str, Any]):
        """Record text characteristics"""
        try:
            # Language detection (simplified)
            text = result.get('original_text', '').lower()
            if any(word in text for word in ['portuguese', 'português', 'brasil', 'brazil']):
                self.language_distribution['portuguese'] += 1
            else:
                self.language_distribution['other'] += 1

            # Domain detection
            intermediate_results = result.get('intermediate_results', {})
            for step_result in intermediate_results.values():
                if isinstance(step_result, dict):
                    context_analysis = step_result.get('context_analysis', {})
                    primary_domain = context_analysis.get('primary_domain')
                    if primary_domain and primary_domain != 'unknown':
                        self.domain_distribution[primary_domain] += 1

            # Entity counts
            entities = result.get('entities', {})
            for entity_type, entity_list in entities.items():
                if isinstance(entity_list, list):
                    self.entity_counts[entity_type] += len(entity_list)

        except Exception as e:
            logger.warning(f"Error recording text characteristics: {str(e)}")

    def _record_hourly_stats(self, result: Dict[str, Any], processing_time: float):
        """Record hourly statistics"""
        try:
            current_hour = datetime.now().strftime('%Y-%m-%d-%H')
            stats = self.hourly_stats[current_hour]

            stats['processed'] += 1
            if not result.get('success', True):
                stats['errors'] += 1

            # Update running averages
            current_count = stats['processed']
            stats['avg_time'] = ((stats['avg_time'] * (current_count - 1)) + processing_time) / current_count

            quality_score = result.get('quality_score', 0.0)
            stats['avg_quality'] = ((stats['avg_quality'] * (current_count - 1)) + quality_score) / current_count

        except Exception as e:
            logger.warning(f"Error recording hourly stats: {str(e)}")

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get comprehensive summary metrics"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time

            # Calculate cache hit rate
            total_cache_requests = self.total_cache_hits + self.total_cache_misses
            cache_hit_rate = (
                self.total_cache_hits / total_cache_requests
                if total_cache_requests > 0 else 0.0
            )

            # Calculate error rate
            error_rate = (
                self.total_errors / self.total_processed
                if self.total_processed > 0 else 0.0
            )

            # Calculate performance statistics
            perf_stats = self._calculate_performance_stats()

            # Calculate quality statistics
            quality_stats = self._calculate_quality_stats()

            summary = {
                'overview': {
                    'total_processed': self.total_processed,
                    'total_errors': self.total_errors,
                    'error_rate': error_rate,
                    'cache_hit_rate': cache_hit_rate,
                    'uptime_seconds': uptime,
                    'avg_processing_time': perf_stats['avg_time'],
                    'avg_quality_score': quality_stats['avg_score']
                },
                'performance': perf_stats,
                'quality': quality_stats,
                'step_metrics': dict(self.step_metrics),
                'distributions': {
                    'languages': dict(self.language_distribution),
                    'domains': dict(self.domain_distribution),
                    'entities': dict(self.entity_counts)
                },
                'errors': {
                    'total_errors': self.total_errors,
                    'error_types': dict(self.error_types),
                    'recent_errors': list(self.error_history)[-10:]  # Last 10 errors
                },
                'hourly_stats': dict(self.hourly_stats),
                'timestamp': datetime.now().isoformat()
            }

            return summary

        except Exception as e:
            logger.error(f"Error generating summary metrics: {str(e)}")
            return {'error': str(e)}

    def _calculate_performance_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics"""
        try:
            if not self.processing_times:
                return {
                    'avg_time': 0.0,
                    'median_time': 0.0,
                    'min_time': 0.0,
                    'max_time': 0.0,
                    'p95_time': 0.0,
                    'throughput_per_second': 0.0
                }

            times_list = list(self.processing_times)
            sorted_times = sorted(times_list)

            # Calculate percentiles
            p95_index = int(len(sorted_times) * 0.95)
            p95_time = sorted_times[min(p95_index, len(sorted_times) - 1)]

            # Calculate throughput
            uptime = time.time() - self.start_time
            throughput_per_second = self.total_processed / uptime if uptime > 0 else 0.0

            return {
                'avg_time': statistics.mean(times_list),
                'median_time': statistics.median(times_list),
                'min_time': min(times_list),
                'max_time': max(times_list),
                'p95_time': p95_time,
                'throughput_per_second': throughput_per_second,
                'total_measurements': len(times_list)
            }

        except Exception as e:
            logger.warning(f"Error calculating performance stats: {str(e)}")
            return {'error': str(e)}

    def _calculate_quality_stats(self) -> Dict[str, Any]:
        """Calculate quality statistics"""
        try:
            if not self.quality_scores:
                return {
                    'avg_score': 0.0,
                    'median_score': 0.0,
                    'min_score': 0.0,
                    'max_score': 0.0,
                    'quality_distribution': {}
                }

            scores_list = list(self.quality_scores)
            sorted_scores = sorted(scores_list)

            # Calculate distribution by quality ranges
            distribution = {
                'excellent': sum(1 for s in scores_list if s >= 0.9),
                'good': sum(1 for s in scores_list if 0.7 <= s < 0.9),
                'fair': sum(1 for s in scores_list if 0.5 <= s < 0.7),
                'poor': sum(1 for s in scores_list if s < 0.5)
            }

            return {
                'avg_score': statistics.mean(scores_list),
                'median_score': statistics.median(scores_list),
                'min_score': min(scores_list),
                'max_score': max(scores_list),
                'quality_distribution': distribution,
                'total_measurements': len(scores_list)
            }

        except Exception as e:
            logger.warning(f"Error calculating quality stats: {str(e)}")
            return {'error': str(e)}

    def get_step_metrics(self, step_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for specific step or all steps"""
        try:
            if step_name:
                return dict(self.step_metrics.get(step_name, {}))
            else:
                return dict(self.step_metrics)

        except Exception as e:
            logger.error(f"Error getting step metrics: {str(e)}")
            return {'error': str(e)}

    def get_quality_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get quality trends over time"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_hour = cutoff_time.strftime('%Y-%m-%d-%H')

            # Filter hourly stats within time range
            recent_stats = {
                hour: stats for hour, stats in self.hourly_stats.items()
                if hour >= cutoff_hour
            }

            # Calculate trends
            if recent_stats:
                hours_list = sorted(recent_stats.keys())
                quality_trend = [recent_stats[hour]['avg_quality'] for hour in hours_list]
                processing_trend = [recent_stats[hour]['processed'] for hour in hours_list]

                return {
                    'time_range_hours': hours,
                    'quality_trend': quality_trend,
                    'processing_trend': processing_trend,
                    'hours': hours_list,
                    'avg_quality_trend': statistics.mean(quality_trend) if quality_trend else 0.0,
                    'total_processed_trend': sum(processing_trend)
                }
            else:
                return {
                    'time_range_hours': hours,
                    'message': 'No data available for the specified time range'
                }

        except Exception as e:
            logger.error(f"Error getting quality trends: {str(e)}")
            return {'error': str(e)}

    def get_error_analysis(self) -> Dict[str, Any]:
        """Get comprehensive error analysis"""
        try:
            error_analysis = {
                'total_errors': self.total_errors,
                'error_rate': self.total_errors / self.total_processed if self.total_processed > 0 else 0.0,
                'error_types': dict(self.error_types),
                'recent_errors': list(self.error_history)[-20:],  # Last 20 errors
                'error_patterns': self._analyze_error_patterns()
            }

            return error_analysis

        except Exception as e:
            logger.error(f"Error getting error analysis: {str(e)}")
            return {'error': str(e)}

    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in errors"""
        try:
            if not self.error_history:
                return {}

            # Analyze error frequency by time of day
            hourly_errors = defaultdict(int)
            for error in self.error_history:
                try:
                    error_time = datetime.fromisoformat(error['timestamp'])
                    hour = error_time.strftime('%H')
                    hourly_errors[hour] += 1
                except:
                    continue

            # Analyze error frequency by text length
            length_errors = defaultdict(int)
            for error in self.error_history:
                text_length = error.get('text_length', 0)
                # Bucket by length ranges
                if text_length < 50:
                    bucket = 'short'
                elif text_length < 200:
                    bucket = 'medium'
                elif text_length < 1000:
                    bucket = 'long'
                else:
                    bucket = 'very_long'
                length_errors[bucket] += 1

            return {
                'errors_by_hour': dict(hourly_errors),
                'errors_by_text_length': dict(length_errors),
                'most_common_error_types': sorted(
                    self.error_types.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }

        except Exception as e:
            logger.warning(f"Error analyzing error patterns: {str(e)}")
            return {}

    def export_metrics(self, filepath: str) -> bool:
        """Export metrics to JSON file"""
        try:
            metrics_data = {
                'summary': self.get_summary_metrics(),
                'export_timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Metrics exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            return False

    def reset_metrics(self):
        """Reset all metrics"""
        try:
            self.total_processed = 0
            self.total_errors = 0
            self.total_cache_hits = 0
            self.total_cache_misses = 0

            self.processing_times.clear()
            self.quality_scores.clear()
            self.text_lengths.clear()

            self.step_metrics.clear()
            self.quality_metrics.clear()
            self.error_types.clear()
            self.error_history.clear()

            self.language_distribution.clear()
            self.domain_distribution.clear()
            self.entity_counts.clear()
            self.hourly_stats.clear()

            self.start_time = time.time()

            logger.info("Preprocessing metrics reset")

        except Exception as e:
            logger.error(f"Error resetting metrics: {str(e)}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the preprocessing system"""
        try:
            current_time = time.time()
            uptime_hours = (current_time - self.start_time) / 3600

            # Calculate health indicators
            error_rate = self.total_errors / self.total_processed if self.total_processed > 0 else 0.0
            avg_quality = statistics.mean(self.quality_scores) if self.quality_scores else 0.0
            avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0.0

            # Determine health status
            if error_rate > 0.1:  # >10% error rate
                status = 'critical'
            elif error_rate > 0.05:  # >5% error rate
                status = 'warning'
            elif avg_quality < 0.5:  # Low quality scores
                status = 'degraded'
            else:
                status = 'healthy'

            health_status = {
                'status': status,
                'uptime_hours': uptime_hours,
                'error_rate': error_rate,
                'average_quality': avg_quality,
                'average_processing_time': avg_processing_time,
                'total_processed': self.total_processed,
                'cache_hit_rate': (
                    self.total_cache_hits / (self.total_cache_hits + self.total_cache_misses)
                    if (self.total_cache_hits + self.total_cache_misses) > 0 else 0.0
                ),
                'indicators': {
                    'high_error_rate': error_rate > 0.05,
                    'low_quality': avg_quality < 0.6,
                    'slow_processing': avg_processing_time > 2.0,  # >2 seconds
                    'low_throughput': self.total_processed / uptime_hours < 10 if uptime_hours > 0 else False
                },
                'timestamp': datetime.now().isoformat()
            }

            return health_status

        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                'status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }