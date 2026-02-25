#!/bin/bash
# Botko Packing Service Manager
# Wrapper for systemctl operations with safety checks

set -euo pipefail

SERVICE_NAME="botko-packing"
PROJECT_DIR="/home/louis/Box-Simulations-botko"
VENV_DIR="$PROJECT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_virtualenv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log_error "Virtual environment not found at $VENV_DIR"
        log_error "Please create it first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        return 1
    fi

    if [[ ! -f "$VENV_PYTHON" ]]; then
        log_error "Python interpreter not found in virtualenv"
        return 1
    fi

    log_info "Virtual environment OK"
    return 0
}

check_service_installed() {
    if ! systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
        log_error "Service not installed. Run scripts/install_service.sh first"
        return 1
    fi
    return 0
}

service_start() {
    log_info "Starting $SERVICE_NAME service..."

    if ! check_virtualenv; then
        return 1
    fi

    if ! check_service_installed; then
        return 1
    fi

    sudo systemctl start "$SERVICE_NAME"

    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Service started successfully"
        service_status
    else
        log_error "Failed to start service"
        return 1
    fi
}

service_stop() {
    log_info "Stopping $SERVICE_NAME service..."

    if ! check_service_installed; then
        return 1
    fi

    sudo systemctl stop "$SERVICE_NAME"
    log_info "Service stopped"
}

service_restart() {
    log_info "Restarting $SERVICE_NAME service..."

    if ! check_virtualenv; then
        return 1
    fi

    if ! check_service_installed; then
        return 1
    fi

    sudo systemctl restart "$SERVICE_NAME"

    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Service restarted successfully"
        service_status
    else
        log_error "Failed to restart service"
        return 1
    fi
}

service_status() {
    if ! check_service_installed; then
        return 1
    fi

    echo ""
    log_info "Service status:"
    systemctl status "$SERVICE_NAME" --no-pager
    echo ""
}

service_logs() {
    local lines="${1:-50}"

    if ! check_service_installed; then
        return 1
    fi

    log_info "Showing last $lines lines of logs (Ctrl+C to exit)..."
    echo ""

    if [[ "$lines" == "follow" ]]; then
        journalctl -u "$SERVICE_NAME" -f
    else
        journalctl -u "$SERVICE_NAME" -n "$lines" --no-pager
    fi
}

service_enable() {
    log_info "Enabling $SERVICE_NAME service to start on boot..."

    if ! check_service_installed; then
        return 1
    fi

    sudo systemctl enable "$SERVICE_NAME"
    log_info "Service enabled"
}

service_disable() {
    log_info "Disabling $SERVICE_NAME service from starting on boot..."

    if ! check_service_installed; then
        return 1
    fi

    sudo systemctl disable "$SERVICE_NAME"
    log_info "Service disabled"
}

show_help() {
    cat << EOF
Botko Packing Service Manager

Usage: $0 <command> [options]

Commands:
    start           Start the service
    stop            Stop the service
    restart         Restart the service
    status          Show service status
    logs [N]        Show last N lines of logs (default: 50)
    logs follow     Follow logs in real-time
    enable          Enable service to start on boot
    disable         Disable service from starting on boot
    help            Show this help message

Examples:
    $0 start
    $0 logs 100
    $0 logs follow
    $0 status

Note: Some commands require sudo privileges.
EOF
}

# Main command router
case "${1:-help}" in
    start)
        service_start
        ;;
    stop)
        service_stop
        ;;
    restart)
        service_restart
        ;;
    status)
        service_status
        ;;
    logs)
        service_logs "${2:-50}"
        ;;
    enable)
        service_enable
        ;;
    disable)
        service_disable
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
