# ============================================================
# Stop & clean up Spark Docker Cluster
# ============================================================
# Usage:
#   .\docker-stop.ps1          → Stop containers
#   .\docker-stop.ps1 -Clean   → Stop + remove volumes & images
# ============================================================

param(
    [switch]$Clean
)

Write-Host ""
Write-Host "Stopping Spark cluster..." -ForegroundColor Yellow
docker compose down

if ($Clean) {
    Write-Host "Removing volumes and images..." -ForegroundColor Yellow
    docker compose down -v --rmi local
    Write-Host "Cleaned up!" -ForegroundColor Green
} else {
    Write-Host "Stopped! (use -Clean to remove images too)" -ForegroundColor Green
}

Write-Host ""
