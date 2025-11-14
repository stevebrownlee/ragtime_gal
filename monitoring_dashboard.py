
"""
Monitoring Dashboard for Feedback-Driven RAG System

This module implements comprehensive monitoring and alerting capabilities
for the RAG system, including real-time metrics collection, performance
tracking, and health monitoring.
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import statistics
import psutil
import sqlite3
from flask import Blueprint, render_template_string, jsonify, request
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_connections: int
    response_time_avg: float
    error_rate: float

@dataclass
class FeedbackMetrics:
    """Feedback system metrics"""
    timestamp: float
    total_feedback: int
    average_rating: float
    rating_distribution: Dict[int, int]
    feedback_rate_per_hour: float
    improvement_trend: float

@dataclass
class Alert:
    """System alert"""
    id: str
    timestamp: float
    level: str  # 'info', 'warning', 'error', 'critical'
    component: str
    message: str
    resolved: bool = False
    resolved_at: Optional[float] = None

class MetricsCollector:
    """Collects and stores system metrics"""

    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=1000)
        self.feedback_buffer = deque(maxlen=1000)
        self.alerts_buffer = deque(maxlen=100)
        self.running = False
        self.collection_thread = None
        self.lock = threading.Lock()

        # Initialize database
        self._init_database()

        # Performance tracking
        self.response_times = deque(maxlen=100)
        self.error_count = 0
        self.request_count = 0
        self.last_reset = time.time()

    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used_mb REAL,
                    disk_usage_percent REAL,
                    active_connections INTEGER,
                    response_time_avg REAL,
                    error_rate REAL
                )
            ''')

            # Feedback metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    total_feedback INTEGER,
                    average_rating REAL,
                    rating_distribution TEXT,
                    feedback_rate_per_hour REAL,
                    improvement_trend REAL
                )
            ''')

            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    level TEXT,
                    component TEXT,
                    message TEXT,
                    resolved BOOLEAN,
                    resolved_at REAL
                )
            ''')

            conn.commit()

    def start_collection(self, interval: int = 30):
        """Start metrics collection in background thread"""
        if self.running:
            return

        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval,),
            daemon=True
        )
        self.collection_thread.start()
        logger.info(f"Started metrics collection with {interval}s interval")

    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped metrics collection")

    def _collection_loop(self, interval: int):
        """Main collection loop"""
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self._store_system_metrics(system_metrics)

                # Collect feedback metrics (if available)
                feedback_metrics = self._collect_feedback_metrics()
                if feedback_metrics:
                    self._store_feedback_metrics(feedback_metrics)

                # Check for alerts
                self._check_alerts(system_metrics)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in metrics collection: {str(e)}")
                time.sleep(interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Network connections (approximate) - requires elevated privileges on macOS
        try:
            connections = len(psutil.net_connections())
        except (psutil.AccessDenied, PermissionError, OSError):
            # Fallback: use process count as rough approximation
            connections = len(psutil.pids())

        # Response time average
        with self.lock:
            avg_response_time = (
                statistics.mean(self.response_times)
                if self.response_times else 0.0
            )

            # Error rate calculation
            current_time = time.time()
            time_window = current_time - self.last_reset
            error_rate = (
                (self.error_count / max(self.request_count, 1)) * 100
                if time_window > 0 else 0.0
            )

            # Reset counters every hour
            if time_window > 3600:
                self.error_count = 0
                self.request_count = 0
                self.last_reset = current_time

        return SystemMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            disk_usage_percent=disk.percent,
            active_connections=connections,
            response_time_avg=avg_response_time,
            error_rate=error_rate
        )

    def _collect_feedback_metrics(self) -> Optional[FeedbackMetrics]:
        """Collect feedback metrics from ConPort or local storage"""
        try:
            current_time = time.time()

            # Placeholder metrics - in real implementation, query ConPort
            total_feedback = 150
            average_rating = 4.2
            rating_distribution = {1: 5, 2: 10, 3: 25, 4: 60, 5: 50}
            feedback_rate_per_hour = 12.5
            improvement_trend = 0.15  # 15% improvement

            return FeedbackMetrics(
                timestamp=current_time,
                total_feedback=total_feedback,
                average_rating=average_rating,
                rating_distribution=rating_distribution,
                feedback_rate_per_hour=feedback_rate_per_hour,
                improvement_trend=improvement_trend
            )

        except Exception as e:
            logger.error(f"Error collecting feedback metrics: {str(e)}")
            return None

    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database"""
        with self.lock:
            self.metrics_buffer.append(metrics)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics
                    (timestamp, cpu_percent, memory_percent, memory_used_mb,
                     disk_usage_percent, active_connections, response_time_avg, error_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
                    metrics.memory_used_mb, metrics.disk_usage_percent,
                    metrics.active_connections, metrics.response_time_avg, metrics.error_rate
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing system metrics: {str(e)}")

    def _store_feedback_metrics(self, metrics: FeedbackMetrics):
        """Store feedback metrics in database"""
        with self.lock:
            self.feedback_buffer.append(metrics)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO feedback_metrics
                    (timestamp, total_feedback, average_rating, rating_distribution,
                     feedback_rate_per_hour, improvement_trend)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp, metrics.total_feedback, metrics.average_rating,
                    json.dumps(metrics.rating_distribution), metrics.feedback_rate_per_hour,
                    metrics.improvement_trend
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing feedback metrics: {str(e)}")

    def _check_alerts(self, metrics: SystemMetrics):
        """Check for alert conditions"""
        alerts = []

        # High CPU usage
        if metrics.cpu_percent > 80:
            alerts.append(Alert(
                id=f"cpu_high_{int(metrics.timestamp)}",
                timestamp=metrics.timestamp,
                level="warning" if metrics.cpu_percent < 90 else "error",
                component="system",
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%"
            ))

        # High memory usage
        if metrics.memory_percent > 85:
            alerts.append(Alert(
                id=f"memory_high_{int(metrics.timestamp)}",
                timestamp=metrics.timestamp,
                level="warning" if metrics.memory_percent < 95 else "critical",
                component="system",
                message=f"High memory usage: {metrics.memory_percent:.1f}%"
            ))

        # High error rate
        if metrics.error_rate > 5:
            alerts.append(Alert(
                id=f"error_rate_high_{int(metrics.timestamp)}",
                timestamp=metrics.timestamp,
                level="error" if metrics.error_rate < 10 else "critical",
                component="application",
                message=f"High error rate: {metrics.error_rate:.1f}%"
            ))

        # Slow response times
        if metrics.response_time_avg > 5:
            alerts.append(Alert(
                id=f"response_slow_{int(metrics.timestamp)}",
                timestamp=metrics.timestamp,
                level="warning" if metrics.response_time_avg < 10 else "error",
                component="application",
                message=f"Slow response times: {metrics.response_time_avg:.2f}s"
            ))

        # Store alerts
        for alert in alerts:
            self._store_alert(alert)

    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        with self.lock:
            self.alerts_buffer.append(alert)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alerts
                    (id, timestamp, level, component, message, resolved, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id, alert.timestamp, alert.level, alert.component,
                    alert.message, alert.resolved, alert.resolved_at
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing alert: {str(e)}")

    def record_request(self, response_time: float, is_error: bool = False):
        """Record request metrics"""
        with self.lock:
            self.response_times.append(response_time)
            self.request_count += 1
            if is_error:
                self.error_count += 1

    def get_recent_metrics(self, hours: int = 24) -> Dict[str, List[Dict]]:
        """Get recent metrics from database"""
        cutoff_time = time.time() - (hours * 3600)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # System metrics
                cursor.execute('''
                    SELECT * FROM system_metrics
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))

                system_metrics = []
                for row in cursor.fetchall():
                    system_metrics.append({
                        'timestamp': row[1],
                        'cpu_percent': row[2],
                        'memory_percent': row[3],
                        'memory_used_mb': row[4],
                        'disk_usage_percent': row[5],
                        'active_connections': row[6],
                        'response_time_avg': row[7],
                        'error_rate': row[8]
                    })

                # Feedback metrics
                cursor.execute('''
                    SELECT * FROM feedback_metrics
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', (cutoff_time,))

                feedback_metrics = []
                for row in cursor.fetchall():
                    feedback_metrics.append({
                        'timestamp': row[1],
                        'total_feedback': row[2],
                        'average_rating': row[3],
                        'rating_distribution': json.loads(row[4]),
                        'feedback_rate_per_hour': row[5],
                        'improvement_trend': row[6]
                    })

                return {
                    'system': system_metrics,
                    'feedback': feedback_metrics
                }

        except Exception as e:
            logger.error(f"Error retrieving metrics: {str(e)}")
            return {'system': [], 'feedback': []}

    def get_active_alerts(self) -> List[Dict]:
        """Get active alerts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM alerts
                    WHERE resolved = 0
                    ORDER BY timestamp DESC
                ''')

                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'level': row[2],
                        'component': row[3],
                        'message': row[4],
                        'resolved': bool(row[5]),
                        'resolved_at': row[6]
                    })

                return alerts

        except Exception as e:
            logger.error(f"Error retrieving alerts: {str(e)}")
            return []

    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE alerts
                    SET resolved = 1, resolved_at = ?
                    WHERE id = ?
                ''', (time.time(), alert_id))
                conn.commit()

        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")

class MonitoringDashboard:
    """Web-based monitoring dashboard"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.blueprint = Blueprint('monitoring', __name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes for dashboard"""

        @self.blueprint.route('/monitoring')
        def dashboard():
            """Main dashboard page"""
            return render_template_string(self._get_dashboard_template())

        @self.blueprint.route('/monitoring/api/metrics')
        def get_metrics():
            """API endpoint for metrics data"""
            hours = request.args.get('hours', 24, type=int)
            metrics = self.metrics_collector.get_recent_metrics(hours)
            return jsonify(metrics)

        @self.blueprint.route('/monitoring/api/alerts')
        def get_alerts():
            """API endpoint for alerts"""
            alerts = self.metrics_collector.get_active_alerts()
            return jsonify(alerts)

        @self.blueprint.route('/monitoring/api/alerts/<alert_id>/resolve', methods=['POST'])
        def resolve_alert(alert_id):
            """API endpoint to resolve alert"""
            self.metrics_collector.resolve_alert(alert_id)
            return jsonify({'status': 'resolved'})

        @self.blueprint.route('/monitoring/api/health')
        def health_check():
            """Health check endpoint"""
            # Get latest metrics
            recent_metrics = self.metrics_collector.get_recent_metrics(hours=1)
            active_alerts = self.metrics_collector.get_active_alerts()

            # Determine overall health
            critical_alerts = [a for a in active_alerts if a['level'] == 'critical']
            error_alerts = [a for a in active_alerts if a['level'] == 'error']

            if critical_alerts:
                status = 'critical'
            elif error_alerts:
                status = 'degraded'
            elif active_alerts:
                status = 'warning'
            else:
                status = 'healthy'

            return jsonify({
                'status': status,
                'timestamp': time.time(),
                'active_alerts': len(active_alerts),
                'critical_alerts': len(critical_alerts),
                'system_metrics_available': len(recent_metrics.get('system', [])) > 0,
                'feedback_metrics_available': len(recent_metrics.get('feedback', [])) > 0
            })

    def _get_dashboard_template(self) -> str:
        """Get HTML template for dashboard"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG System Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .alerts-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .alert {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert.critical { border-color: #dc3545; background-color: #f8d7da; }
        .alert.error { border-color: #fd7e14; background-color: #fff3cd; }
        .alert.warning { border-color: #ffc107; background-color: #fff3cd; }
        .alert.info { border-color: #17a2b8; background-color: #d1ecf1; }
        .resolve-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            float: right;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-healthy { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #fd7e14; }
        .status-critical { background-color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RAG System Monitoring Dashboard</h1>
        <p>Real-time monitoring and performance metrics</p>
        <div id="system-status">
            <span class="status-indicator status-healthy"></span>
            <span>System Status: Loading...</span>
        </div>
    </div>

    <div class="metrics-grid" id="metrics-grid">
        <!-- Metrics cards will be populated by JavaScript -->
    </div>

    <div class="chart-container">
        <h3>System Performance (Last 24 Hours)</h3>
        <canvas id="performance-chart" width="400" height="200"></canvas>
    </div>

    <div class="chart-container">
        <h3>Feedback Metrics</h3>
        <canvas id="feedback-chart" width="400" height="200"></canvas>
    </div>

    <div class="alerts-container">
        <h3>Active Alerts</h3>
        <div id="alerts-list">
            <!-- Alerts will be populated by JavaScript -->
        </div>
    </div>

    <script>
        // Global variables
        let performanceChart;
        let feedbackChart;

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadDashboard();
            setInterval(loadDashboard, 30000); // Refresh every 30 seconds
        });

        async function loadDashboard() {
            try {
                await loadHealthStatus();
                await loadMetrics();
                await loadAlerts();
            } catch (error) {
                console.error('Error loading dashboard:', error);
            }
        }

        async function loadHealthStatus() {
            try {
                const response = await fetch('/monitoring/api/health');
                const health = await response.json();

                const statusElement = document.getElementById('system-status');
                const indicator = statusElement.querySelector('.status-indicator');
                const text = statusElement.querySelector('span:last-child');

                indicator.className = `status-indicator status-${health.status}`;
                text.textContent = `System Status: ${health.status.toUpperCase()}`;

            } catch (error) {
                console.error('Error loading health status:', error);
            }
        }

        async function loadMetrics() {
            try {
                const response = await fetch('/monitoring/api/metrics?hours=24');
                const data = await response.json();

                updateMetricsCards(data);
                updatePerformanceChart(data.system);
                updateFeedbackChart(data.feedback);

            } catch (error) {
                console.error('Error loading metrics:', error);
            }
        }

        function updateMetricsCards(data) {
            const systemMetrics = data.system;
            const feedbackMetrics = data.feedback;

            if (systemMetrics.length === 0) return;

            const latest = systemMetrics[0];
            const latestFeedback = feedbackMetrics.length > 0 ? feedbackMetrics[0] : null;

            const metricsGrid = document.getElementById('metrics-grid');
            metricsGrid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${latest.cpu_percent.toFixed(1)}%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${latest.memory_percent.toFixed(1)}%</div>
                    <div class="metric-label">Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${latest.response_time_avg.toFixed(2)}s</div>
                    <div class="metric-label">Avg Response Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${latest.error_rate.toFixed(1)}%</div>
                    <div class="metric-label">Error Rate</div>
                </div>
                ${latestFeedback ? `
                <div class="metric-card">
                    <div class="metric-value">${latestFeedback.average_rating.toFixed(1)}/5</div>
                    <div class="metric-label">Average Rating</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${latestFeedback.total_feedback}</div>
                    <div class="metric-label">Total Feedback</div>
                </div>
                ` : ''}
            `;
        }

        function updatePerformanceChart(systemMetrics) {
            if (systemMetrics.length === 0) return;

            const ctx = document.getElementById('performance-chart').getContext('2d');

            const labels = systemMetrics.reverse().map(m =>
                new Date(m.timestamp * 1000).toLocaleTimeString()
            );

            const cpuData = systemMetrics.map(m => m.cpu_percent);
            const memoryData = systemMetrics.map(m => m.memory_percent);
            const responseTimeData = systemMetrics.map(m => m.response_time_avg * 100);

            if (performanceChart) {
                performanceChart.destroy();
            }

            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'CPU %',
                            data: cpuData,
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            tension: 0.1
                        },
                        {
                            label: 'Memory %',
                            data: memoryData,
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            tension: 0.1
                        },
                        {
                            label: 'Response Time (x100ms)',
                            data: responseTimeData,
                            borderColor: 'rgb(255, 205, 86)',
                            backgroundColor: 'rgba(255, 205, 86, 0.1)',
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }

        function updateFeedbackChart(feedbackMetrics) {
            if (feedbackMetrics.length === 0) return;

            const ctx = document.getElementById('feedback-chart').getContext('2d');
            const latest = feedbackMetrics[0];

            if (feedbackChart) {
                feedbackChart.destroy();
            }

            feedbackChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars'],
                    datasets: [{
                        data: [
                            latest.rating_distribution[1] || 0,
                            latest.rating_distribution[2] || 0,
                            latest.rating_distribution[3] || 0,
                            latest.rating_distribution[4] || 0,
                            latest.rating_distribution[5] || 0
                        ],
                        backgroundColor: [
                            '#dc3545',
                            '#fd7e14',
                            '#ffc107',
                            '#28a745',
                            '#20c997'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }

        async function loadAlerts() {
            try {
                const response = await fetch('/monitoring/api/alerts');
                const alerts = await response.json();

                const alertsList = document.getElementById('alerts-list');

                if (alerts.length === 0) {
                    alertsList.innerHTML = '<p>No active alerts</p>';
                    return;
                }

                alertsList.innerHTML = alerts.map(alert => `
                    <div class="alert ${alert.level}">
                        <strong>${alert.component.toUpperCase()}</strong>: ${alert.message}
                        <button class="resolve-btn" onclick="resolveAlert('${alert.id}')">
                            Resolve
                        </button>
                        <br>
                        <small>${new Date(alert.timestamp * 1000).toLocaleString()}</small>
                    </div>
                `).join('');

            } catch (error) {
                console.error('Error loading alerts:', error);
            }
        }

        async function resolveAlert(alertId) {
            try {
                const response = await fetch(`/monitoring/api/alerts/${alertId}/resolve`, {
                    method: 'POST'
                });

                if (response.ok) {
                    await loadAlerts();
                }

            } catch (error) {
                console.error('Error resolving alert:', error);
            }
        }
    </script>
</body>
</html>
        """

# Global metrics collector instance
metrics_collector = MetricsCollector()

def create_monitoring_blueprint() -> Blueprint:
    """Create and return monitoring blueprint"""
    dashboard = MonitoringDashboard(metrics_collector)
    return dashboard.blueprint

def start_monitoring(interval: int = 30):
    """Start monitoring system"""
    metrics_collector.start_collection(interval)

def stop_monitoring():
    """Stop monitoring system"""
    metrics_collector.stop_collection()

def record_request_metrics(response_time: float, is_error: bool = False):
    """Record request metrics for monitoring"""
    metrics_collector.record_request(response_time, is_error)

# Context manager for request timing
@contextmanager
def monitor_request():
    """Context manager to automatically monitor request performance"""
    start_time = time.time()
    error_occurred = False

    try:
        yield
    except Exception as e:
        error_occurred = True
        raise
    finally:
        response_time = time.time() - start_time
        record_request_metrics(response_time, error_occurred)

# Decorator for monitoring Flask routes
def monitor_route(func):
    """Decorator to monitor Flask route performance"""
    def wrapper(*args, **kwargs):
        with monitor_request():
            return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

if __name__ == "__main__":
    # Test the monitoring system
    import random

    def simulate_requests():
        """Simulate requests for testing"""
        while True:
            # Simulate request with random response time and occasional errors
            response_time = random.uniform(0.1, 3.0)
            is_error = random.random() < 0.05  # 5% error rate

            record_request_metrics(response_time, is_error)
            time.sleep(random.uniform(1, 5))

    # Start monitoring
    start_monitoring(interval=10)

    # Start request simulation
    simulation_thread = threading.Thread(target=simulate_requests, daemon=True)
    simulation_thread.start()

    print("Monitoring system started. Check the dashboard at /monitoring")
    print("Press Ctrl+C to stop...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitoring system...")
        stop_monitoring()