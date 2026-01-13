# Deployment Guide for Linux Server

This guide covers deploying the Pipeline Testing App to a Linux server.

## Prerequisites

- Linux server (Ubuntu 20.04+ recommended)
- Python 3.10 or higher
- Neo4j database accessible from the server
- (Optional) NVIDIA GPU with CUDA for faster embeddings

## Directory Structure

```
/opt/pipeline-tester/
├── app.py
├── config.py
├── database.py
├── requirements.txt
├── .env
├── .streamlit/
│   └── config.toml
├── pipelines/
│   ├── __init__.py
│   ├── base.py
│   └── pipeline_v4.py
├── components/
│   ├── __init__.py
│   ├── answer_display.py
│   ├── context_display.py
│   ├── trace_display.py
│   ├── agent_graph_display.py
│   ├── history_display.py
│   └── batch_display.py
├── data/
│   └── app.db (created automatically)
└── deploy/
    ├── README.md
    ├── streamlit.service
    └── nginx.conf
```

## Installation Steps

### 1. Clone/Copy Files

```bash
# Create directory
sudo mkdir -p /opt/pipeline-tester
sudo chown $USER:$USER /opt/pipeline-tester

# Copy all files to the server
# scp -r StreamlitApp/* user@server:/opt/pipeline-tester/
```

### 2. Create Virtual Environment

```bash
cd /opt/pipeline-tester

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env
nano .env
```

Fill in all required values:
- `QWEN_API_KEY`: Your Qwen API key
- `NEO4J_URI`: Neo4j connection URI (e.g., bolt://neo4j-server:7687)
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `LANGFUSE_PUBLIC_KEY`: Langfuse public key
- `LANGFUSE_SECRET_KEY`: Langfuse secret key
- `APP_SECRET_KEY`: Generate a random secret key

### 4. Test Locally

```bash
source venv/bin/activate
streamlit run app.py --server.port 8501
```

Visit `http://server-ip:8501` to verify it works.

### 5. Set Up Systemd Service

```bash
# Copy service file
sudo cp deploy/streamlit.service /etc/systemd/system/

# Edit if needed (update paths, user)
sudo nano /etc/systemd/system/streamlit.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit

# Check status
sudo systemctl status streamlit
```

### 6. Configure Nginx Reverse Proxy

```bash
# Copy nginx config
sudo cp deploy/nginx.conf /etc/nginx/sites-available/pipeline-tester

# Edit domain/SSL settings
sudo nano /etc/nginx/sites-available/pipeline-tester

# Enable site
sudo ln -s /etc/nginx/sites-available/pipeline-tester /etc/nginx/sites-enabled/

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

### 7. SSL Certificate (Optional but Recommended)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up automatically
```

## Troubleshooting

### Check Logs

```bash
# Streamlit logs
sudo journalctl -u streamlit -f

# Nginx logs
sudo tail -f /var/log/nginx/error.log
```

### Common Issues

1. **Port 8501 in use**: Check for other processes: `sudo lsof -i :8501`

2. **Neo4j connection failed**: Verify Neo4j is accessible from server

3. **GPU not detected**: Install CUDA toolkit and restart

4. **Import errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

5. **Permission denied**: Check file permissions and ownership

## Updating

```bash
# Stop service
sudo systemctl stop streamlit

# Update files
# scp -r StreamlitApp/* user@server:/opt/pipeline-tester/

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Start service
sudo systemctl start streamlit
```

## Monitoring

You can set up monitoring with:
- **Uptime checks**: Use Uptime Robot or similar
- **Log aggregation**: Use journald or ship to ELK stack
- **Metrics**: Streamlit has built-in metrics at /_stcore/health
