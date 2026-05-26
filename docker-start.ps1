# ============================================================
# Start Spark Distributed Cluster (Docker)
# ============================================================
# Usage:
#   .\docker-start.ps1              → Start cluster only
#   .\docker-start.ps1 -Run         → Start cluster + run pipelines
#   .\docker-start.ps1 -Build       → Rebuild images + start
#   .\docker-start.ps1 -Build -Run  → Rebuild + start + run
# ============================================================

param(
    [switch]$Build,
    [switch]$Run
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Spark Docker Cluster Manager" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Build images if requested
if ($Build) {
    Write-Host "[1/3] Building Docker image..." -ForegroundColor Yellow
    docker compose build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed!"
        exit 1
    }
    Write-Host "  Image built successfully!" -ForegroundColor Green
} else {
    Write-Host "[1/3] Skipping build (use -Build flag to rebuild)" -ForegroundColor DarkGray
}

# Step 2: Start cluster (Master + 2 Workers)
Write-Host "[2/3] Starting Spark cluster..." -ForegroundColor Yellow
docker compose up -d spark-master spark-worker-1 spark-worker-2

Start-Sleep -Seconds 5

# Check cluster status
Write-Host ""
Write-Host "  Cluster Status:" -ForegroundColor Cyan
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

Write-Host ""
Write-Host "  Master UI:   http://localhost:8080" -ForegroundColor Green
Write-Host "  Worker 1 UI: http://localhost:8081" -ForegroundColor Green
Write-Host "  Worker 2 UI: http://localhost:8082" -ForegroundColor Green

# Step 3: Run pipelines if requested
if ($Run) {
    Write-Host ""
    Write-Host "[3/3] Running pipelines..." -ForegroundColor Yellow
    Write-Host "  App UI will be at: http://localhost:4040" -ForegroundColor Green
    Write-Host ""
    docker compose up spark-submit
} else {
    Write-Host ""
    Write-Host "[3/3] Cluster is ready! To run pipelines:" -ForegroundColor Yellow
    Write-Host '  docker compose up spark-submit' -ForegroundColor White
    Write-Host ""
    Write-Host "  Or run a single pipeline:" -ForegroundColor Yellow
    Write-Host '  docker compose run --rm spark-submit /opt/spark/bin/spark-submit --master spark://spark-master:7077 /app/pipeline_01_eda.py' -ForegroundColor White
}

Write-Host ""
