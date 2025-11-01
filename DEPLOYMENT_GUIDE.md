
# Deployment Guide: Feedback-Driven RAG System

## Table of Contents
1. [Overview](#overview)
2. [Production Environment Setup](#production-environment-setup)
3. [Configuration Management](#configuration-management)
4. [Database Initialization](#database-initialization)
5. [Containerized Deployment](#containerized-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Security Configuration](#security-configuration)
8. [Performance Tuning](#performance-tuning)
9. [Backup and Recovery](#backup-and-recovery)
10. [Maintenance Procedures](#maintenance-procedures)
11. [Troubleshooting](#troubleshooting)

## Overview

This guide provides comprehensive instructions for deploying the Feedback-Driven RAG System in production environments. The system supports multiple deployment strategies including containerized deployment with Docker, cloud deployment, and traditional server deployment.

### Deployment Options

1. **Docker Compose** (Recommended for most use cases)
2. **Kubernetes** (For large-scale deployments)
3. **Traditional Server** (For on-premises deployments)
4. **Cloud Platforms** (AWS, GCP, Azure)

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 100 Mbps

**Recommended Requirements:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- Network: 1 Gbps
- GPU: Optional (for model fine-tuning)

## Production Environment Setup

### 1. Server Preparation

**Ubuntu/Debian:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip python3-venv git curl wget

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Nginx (for reverse proxy)
sudo apt install -y nginx

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

**CentOS/RHEL:**
```bash
# Update system
sudo yum update -y

# Install required packages
sudo yum install -y python3 python3-pip git curl wget

# Install Docker
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Nginx
sudo yum install -y nginx

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. User and Directory Setup

```bash
# Create application user
sudo useradd -m -s /bin/bash ragtime
sudo usermod -aG docker ragtime

# Create application directories
sudo mkdir -p /opt/ragtime-gal
sudo mkdir -p /var/log/ragtime-gal
sudo mkdir -p /var/lib/ragtime-gal/data
sudo mkdir -p /var/lib/ragtime-gal/models

# Set permissions
sudo chown -R ragtime:ragtime /opt/ragtime-gal
sudo chown -R ragtime:ragtime /var/log/ragtime-gal
sudo chown -R ragtime:ragtime /var/lib/ragtime-gal
```

### 3. Application Deployment

```bash
# Switch to application user
sudo su - ragtime

# Clone repository
cd /opt/ragtime-gal
git clone <repository-url> .

# Create production environment file
cp .env.template .env.production
```

## Configuration Management

### 1. Environment Variables

Create `/opt/ragtime-gal/.env.production`:

```bash
# Application Configuration
FLASK_ENV=production
DEBUG=false
SECRET_KEY=your-super-secret-key-here
PORT=8084

# Database Configuration
CHROMA_PERSIST_DIR=/var/lib/ragtime-gal/data/chroma_db
TEMP_FOLDER=/var/lib/ragtime-gal/temp

# LLM Configuration
LLM_MODEL=sixthwood
EMBEDDING_MODEL=mistral
OLLAMA_BASE_URL=http://localhost:11434
RETRIEVAL_K=4
TEMPERATURE=0.7
MAX_TOKENS=500

# Monitoring Configuration
MONITORING_DB_PATH=/var/lib/ragtime-gal/data/monitoring.db
METRICS_COLLECTION_INTERVAL=30
LOG_LEVEL=INFO

# Security Configuration
ALLOWED_HOSTS=your-domain.com,localhost
CORS_ORIGINS=https://your-domain.com
SESSION_TIMEOUT=3600

# Performance Configuration
WORKERS=4
WORKER_CONNECTIONS=1000
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=100

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE=0 2 * * *  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
BACKUP_LOCATION=/var/lib/ragtime-gal/backups
```

### 2. Logging Configuration

Create `/opt/ragtime-gal/logging.conf`:

```ini
[loggers]
keys=root,app,monitoring

[handlers]
keys=consoleHandler,fileHandler,errorHandler

[formatters]
keys=simpleFormatter,detailedFormatter

[logger_root]
level=INFO
handlers=consoleHandler,fileHandler

[logger_app]
level=INFO
handlers=fileHandler,errorHandler
qualname=app
propagate=0

[logger_monitoring]
level=INFO
handlers=fileHandler
qualname=monitoring
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=handlers.RotatingFileHandler
level=INFO
formatter=detailedFormatter
args=('/var/log/ragtime-gal/app.log', 'a', 10485760, 10)

[handler_errorHandler]
class=handlers.RotatingFileHandler
level=ERROR
formatter=detailedFormatter
args=('/var/log/ragtime-gal/error.log', 'a', 10485760, 10)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s
```

## Database Initialization

### 1. ChromaDB Setup

```bash
# Create ChromaDB directory
sudo mkdir -p /var/lib/ragtime-gal/data/chroma_db
sudo chown -R ragtime:ragtime /var/lib/ragtime-gal/data/chroma_db

# Initialize database (run as ragtime user)
cd /opt/ragtime-gal
python3 -c "
import os
os.environ['CHROMA_PERSIST_DIR'] = '/var/lib/ragtime-gal/data/chroma_db'
from app import get_vector_db
db = get_vector_db()
print('ChromaDB initialized successfully')
"
```

### 2. Monitoring Database Setup

```bash
# Initialize monitoring database
python3 -c "
import os
os.environ['MONITORING_DB_PATH'] = '/var/lib/ragtime-gal/data/monitoring.db'
from monitoring_dashboard import MetricsCollector
collector = MetricsCollector('/var/lib/ragtime-gal/data/monitoring.db')
print('Monitoring database initialized successfully')
"
```

## Containerized Deployment

### 1. Docker Compose Production Setup

Create `/opt/ragtime-gal/docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  ragtime-app:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: ragtime-app
    restart: unless-stopped
    ports:
      - "8084:8084"
    environment:
      - FLASK_ENV=production
      - DEBUG=false
    env_file:
      - .env.production
    volumes:
      - /var/lib/ragtime-gal/data:/app/data
      - /var/log/ragtime-gal:/app/logs
      - /var/lib/ragtime-gal/models:/app/models
    depends_on:
      - ollama
    networks:
      - ragtime-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8084/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  ollama:
    image: ollama/ollama:latest
    container_name: ragtime-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - /var/lib/ragtime-gal/models:/root/.ollama
    networks:
      - ragtime-network
    environment:
      - OLLAMA_HOST=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: ragtime-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - /var/log/nginx:/var/log/nginx
    depends_on:
      - ragtime-app
    networks:
      - ragtime-network

  prometheus:
    image: prom/prometheus:latest
    container_name: ragtime-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - ragtime-network

  grafana:
    image: grafana/grafana:latest
    container_name: ragtime-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - ragtime-network

networks:
  ragtime-network:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
```

### 2. Production Dockerfile

Create `/opt/ragtime-gal/Dockerfile.prod`:

```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn for production
RUN pip install gunicorn

# Copy project
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models

# Expose port
EXPOSE 8084

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8084/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8084", "--workers", "4", "--timeout", "120", "--keep-alive", "2", "--max-requests", "1000", "--max-requests-jitter", "100", "app:app"]
```

### 3. Nginx Configuration

Create `/opt/ragtime-gal/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream ragtime_app {
        server ragtime-app:8084;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # File upload size
        client_max_body_size 100M;

        # Gzip compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://ragtime_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Upload endpoints with stricter rate limiting
        location /embed {
            limit_req zone=upload burst=5 nodelay;
            proxy_pass http://ragtime_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # Monitoring dashboard
        location /monitoring {
            auth_basic "Monitoring Dashboard";
            auth_basic_user_file /etc/nginx/.htpasswd;
            proxy_pass http://ragtime_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Main application
        location / {
            proxy_pass http://ragtime_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Health check
        location /health {
            proxy_pass http://ragtime_app;
            access_log off;
        }
    }
}
```

### 4. Deployment Commands

```bash
# Build and start services
cd /opt/ragtime-gal
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Initialize Ollama models
docker exec ragtime-ollama ollama pull mistral
docker exec ragtime-ollama ollama pull sixthwood

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f ragtime-app
```

## Monitoring Setup

### 1. Prometheus Configuration

Create `/opt/ragtime-gal/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'ragtime-app'
    static_configs:
      - targets: ['ragtime-app:8084']
    metrics_path: '/monitoring/api/metrics'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    metrics_path: '/metrics'
```

### 2. Alert Rules

Create `/opt/ragtime-gal/alert_rules.yml`:

```yaml
groups:
  - name: ragtime_alerts
    rules:
      - alert: HighCPUUsage
        expr: cpu_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: memory_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes"

      - alert: HighErrorRate
        expr: error_rate > 5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% for more than 2 minutes"

      - alert: SlowResponseTime
        expr: response_time_avg > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow response times detected"
          description: "Average response time is above 10 seconds"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"
```

### 3. Grafana Dashboard

Create `/opt/ragtime-gal/grafana/provisioning/dashboards/ragtime-dashboard.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "RAG Time Gal System Dashboard",
    "tags": ["ragtime", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "System Metrics",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "memory_percent",
            "legendFormat": "Memory %"
          }
        ],
        "yAxes": [
          {
            "min": 0,
            "max": 100,
            "unit": "percent"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Times",
        "type": "graph",
        "targets": [
          {
            "expr": "response_time_avg",
            "legendFormat": "Avg Response Time"
          }
        ],
        "yAxes": [
          {
            "min": 0,
            "unit": "s"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "error_rate",
            "legendFormat": "Error Rate"
          }
        ],
        "valueName": "current",
        "unit": "percent",
        "thresholds": "5,10"
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## Security Configuration

### 1. SSL Certificate Setup

```bash
# Using Let's Encrypt (recommended)
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet

# Or using self-signed certificate (development only)
sudo mkdir -p /opt/ragtime-gal/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /opt/ragtime-gal/ssl/key.pem \
    -out /opt/ragtime-gal/ssl/cert.pem
```

### 2. Firewall Configuration

```bash
# UFW (Ubuntu/Debian)
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 8084/tcp  # Block direct access to app
sudo ufw deny 11434/tcp  # Block direct access to Ollama

# iptables (CentOS/RHEL)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --reload
```

### 3. Authentication Setup

```bash
# Create monitoring dashboard password
sudo htpasswd -c /etc/nginx/.htpasswd admin

# Set secure permissions
sudo chmod 600 /etc/nginx/.htpasswd
```

## Performance Tuning

### 1. System Optimization

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize kernel parameters
echo "net.core.somaxconn = 65536" | sudo tee -a /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" | sudo tee -a /etc/sysctl.conf
echo "vm.swappiness = 10" | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p
```

### 2. Application Optimization

Update `.env.production`:

```bash
# Worker configuration
WORKERS=4  # Number of CPU cores
WORKER_CONNECTIONS=1000
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=100

# Database optimization
CHROMA_BATCH_SIZE=100
EMBEDDING_CACHE_SIZE=1000

# Memory optimization
PYTHON_GC_THRESHOLD=700,10,10
```

### 3. Database Optimization

```bash
# ChromaDB optimization
echo "PRAGMA journal_mode=WAL;" | sqlite3 /var/lib/ragtime-gal/data/chroma_db/chroma.sqlite3
echo "PRAGMA synchronous=NORMAL;" | sqlite3 /var/lib/ragtime-gal/data/chroma_db/chroma.sqlite3
echo "PRAGMA cache_size=10000;" | sqlite3 /var/lib/ragtime-gal/data/chroma_db/chroma.sqlite3
```

## Backup and Recovery

### 1. Backup Script

Create `/opt/ragtime-gal/scripts/backup.sh`:

```bash
#!/bin/bash

# Configuration
BACKUP_DIR="/var/lib/ragtime-gal/backups"
DATA_DIR="/var/lib/ragtime-gal/data"
RETENTION_DAYS=30
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="ragtime_backup_${DATE}"

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Stop services for consistent backup
echo "Stopping services..."
docker-compose -f /opt/ragtime-gal/docker-compose.prod.yml stop ragtime-app

# Create backup
echo "Creating backup: ${BACKUP_NAME}"
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
    -C "${DATA_DIR}" \
    chroma_db \
    monitoring.db \
    --exclude="*.tmp" \
    --exclude="*.log"

# Restart services
echo "Starting services..."
docker-compose -f /opt/ragtime-gal/docker-compose.prod.yml start ragtime-app

# Clean old backups
echo "Cleaning old backups..."
find "${BACKUP_DIR}" -name "ragtime_backup_*.tar.gz" -mtime +${RETENTION_DAYS} -delete

# Verify backup
if [ -f "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" ]; then
    echo "Backup completed successfully: ${BACKUP_NAME}.tar.gz"
    echo "Backup size: $(du -h "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)"
else
    echo "Backup failed!"
    exit 1
fi
```

### 2. Automated Backup Setup

```bash
# Make script executable
chmod +x /opt/ragtime-gal/scripts/backup.sh

# Add to crontab
sudo crontab -e -u ragtime
# Add: 0 2 * * * /opt/ragtime-gal/scripts/backup.sh >> /var/log/ragtime-gal/backup.log 2>&1
```

### 3. Recovery Procedure

Create `/opt/ragtime-gal/scripts/restore.sh`:

```bash
#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <backup_file>"
    echo "Available backups:"
    ls -la /var/lib/ragtime-gal/backups/ragtime_backup_*.tar.gz
    exit 1
fi

BACKUP_FILE="$1"
DATA_DIR="/var/lib/ragtime-gal/data"

# Verify backup file exists
if [ ! -f "${BACKUP_FILE}" ]; then
    echo "Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

# Stop services
echo "Stopping services..."
docker-compose -f /opt/ragtime-gal/docker-compose.prod.yml stop

# Backup current data
echo "Backing up current data..."
mv "${DATA_DIR}" "${DATA_DIR}.backup.$(date +%Y%m%d_%H%M%S)"

# Create new data directory
mkdir -p "${DATA_DIR}"

# Restore from backup
echo "Restoring from backup..."
tar -xzf "${BACKUP_FILE}" -C "${DATA_DIR}"

# Set permissions
chown -R ragtime:ragtime "${DATA_DIR}"

# Start services
echo "Starting services..."
docker-compose -f /opt/ragtime-gal/docker-compose.prod.yml start

echo "Restore completed successfully!"
```

## Maintenance Procedures

### 1. Regular Maintenance Tasks

Create `/opt/ragtime-gal/scripts/maintenance.sh`:

```bash
#!/bin/bash

echo "Starting maintenance tasks..."

# Clean temporary files
echo "Cleaning temporary files..."
find /var/lib/ragtime-gal/temp -type f -mtime +7 -delete

# Rotate logs
echo "Rotating logs..."
find /var/log/ragtime-gal -name "*.log" -size +100M -exec logrotate {} \;

# Update Docker images
echo "Updating Docker images..."
docker-compose -f /opt/ragtime-gal/docker-compose.prod.yml pull

# Clean unused Docker resources
echo "Cleaning Docker resources..."
docker system prune -f

# Vacuum databases
echo "Optimizing databases..."
sqlite3 /var/lib/ragtime-gal/data/monitoring.db "VACUUM;"

# Check disk space
echo "Checking disk space..."
df -h /var/lib/ragtime-gal

echo "Maintenance completed!"
```

### 2. Health Check Script

Create `/opt/ragtime-gal/scripts/health_check.sh`:

```bash
#!/bin/bash

# Configuration
APP_URL="https://your-domain.com"
MONITORING_URL="${APP_URL}/monitoring/api/health"
ALERT_EMAIL="admin@your-domain.com"

# Check application health
echo "Checking application health..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${APP_URL}/health")

if [ "${HTTP_STATUS}" != "200" ]; then
    echo "Application health check failed! HTTP Status: ${HTTP_STATUS}"
    echo "Application is down!" | mail -s "RAG System Alert" "${ALERT_EMAIL}"
    exit 1
fi

# Check monitoring health
echo "Checking monitoring health..."
MONITORING_STATUS=$(curl -s "${MONITORING_URL}" | jq -r '.status')

if [ "${MONITORING_STATUS}" != "healthy" ]; then
    echo "System status: ${MONITORING_STATUS}"
    echo "System health degraded: ${MONITORING_STATUS}" | mail -s "RAG System Warning" "${ALERT_EMAIL}"
fi

# Check disk space
DISK_USAGE=$(df /var/lib/ragtime-gal | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "${DISK_USAGE}" -gt 80 ]; then
    echo "High disk usage: ${DISK_USAGE}%"
    echo "High disk usage detected: ${DISK_USAGE}%" | mail -s "RAG System Warning" "${ALERT_EMAIL}"
fi

echo "Health check completed successfully!"
```

### 3. Update Procedure

Create `/opt/ragtime-gal/scripts/update.sh`:

```bash
#!/bin/bash

echo "Starting system update..."

# Backup current version
echo "Creating backup..."
/opt