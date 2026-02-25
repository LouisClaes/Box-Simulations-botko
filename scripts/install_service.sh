#!/bin/bash
# Botko Packing Service Installation Script
# Installs systemd service for overnight experiments

set -euo pipefail

SERVICE_NAME="botko-packing"
PROJECT_DIR="/home/louis/Box-Simulations-botko"
SERVICE_FILE="$PROJECT_DIR/scripts/$SERVICE_NAME.service"
SYSTEMD_DIR="/etc/systemd/system"
INSTALLED_SERVICE="$SYSTEMD_DIR/$SERVICE_NAME.service"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

check_prerequisites() {
    log_step "Checking prerequisites..."

    # Check if running as root or can use sudo
    if [[ $EUID -eq 0 ]]; then
        log_warn "Running as root. Using direct commands."
        SUDO_CMD=""
    elif command -v sudo &> /dev/null; then
        log_info "Using sudo for privileged operations"
        SUDO_CMD="sudo"
    else
        log_error "This script requires root privileges or sudo"
        exit 1
    fi

    # Check if service file exists
    if [[ ! -f "$SERVICE_FILE" ]]; then
        log_error "Service file not found: $SERVICE_FILE"
        exit 1
    fi

    # Check if virtualenv exists
    if [[ ! -d "$PROJECT_DIR/venv" ]]; then
        log_error "Virtual environment not found at $PROJECT_DIR/venv"
        log_error "Please create it first:"
        echo "  cd $PROJECT_DIR"
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
        exit 1
    fi

    # Check if experiment module exists
    if [[ ! -f "$PROJECT_DIR/src/runner/__init__.py" ]]; then
        log_warn "src/runner module structure exists but may need experiment.py"
    fi

    log_info "Prerequisites OK"
}

backup_existing_service() {
    if [[ -f "$INSTALLED_SERVICE" ]]; then
        log_step "Backing up existing service file..."
        local backup="$INSTALLED_SERVICE.backup.$(date +%Y%m%d_%H%M%S)"
        $SUDO_CMD cp "$INSTALLED_SERVICE" "$backup"
        log_info "Backup created: $backup"

        # Stop service if running
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            log_step "Stopping existing service..."
            $SUDO_CMD systemctl stop "$SERVICE_NAME"
        fi
    fi
}

install_service() {
    log_step "Installing service file..."

    # Copy service file to systemd directory
    $SUDO_CMD cp "$SERVICE_FILE" "$INSTALLED_SERVICE"

    # Set correct permissions
    $SUDO_CMD chmod 644 "$INSTALLED_SERVICE"

    log_info "Service file installed to $INSTALLED_SERVICE"
}

reload_systemd() {
    log_step "Reloading systemd daemon..."
    $SUDO_CMD systemctl daemon-reload
    log_info "Systemd daemon reloaded"
}

enable_service() {
    log_step "Enabling service..."
    $SUDO_CMD systemctl enable "$SERVICE_NAME"
    log_info "Service enabled (will start on boot)"
}

show_status() {
    log_step "Service configuration complete!"
    echo ""
    log_info "Service: $SERVICE_NAME"
    log_info "Status: Installed and enabled"
    log_info "User: louis"
    log_info "Working directory: $PROJECT_DIR"
    log_info "CPU quota: 50%"
    log_info "Nice level: 10"
    echo ""
    log_info "Next steps:"
    echo "  1. Start the service:    scripts/service_manager.sh start"
    echo "  2. Check status:         scripts/service_manager.sh status"
    echo "  3. View logs:            scripts/service_manager.sh logs"
    echo "  4. Follow logs:          scripts/service_manager.sh logs follow"
    echo ""
    log_warn "Note: Make sure src.runner.experiment module is ready before starting"
}

uninstall_service() {
    log_step "Uninstalling service..."

    # Stop service if running
    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        log_step "Stopping service..."
        $SUDO_CMD systemctl stop "$SERVICE_NAME"
    fi

    # Disable service if enabled
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        log_step "Disabling service..."
        $SUDO_CMD systemctl disable "$SERVICE_NAME"
    fi

    # Remove service file
    if [[ -f "$INSTALLED_SERVICE" ]]; then
        $SUDO_CMD rm "$INSTALLED_SERVICE"
        log_info "Service file removed"
    fi

    # Reload systemd
    $SUDO_CMD systemctl daemon-reload

    log_info "Service uninstalled successfully"
}

show_help() {
    cat << EOF
Botko Packing Service Installation Script

Usage: $0 [command]

Commands:
    install         Install and enable the systemd service (default)
    uninstall       Remove the systemd service
    reinstall       Uninstall and reinstall the service
    help            Show this help message

Examples:
    $0                  # Install service
    $0 install          # Install service
    $0 uninstall        # Remove service
    $0 reinstall        # Reinstall service

The service will be installed to: $SYSTEMD_DIR/$SERVICE_NAME.service

Note: This script requires sudo privileges.
EOF
}

# Main installation flow
main_install() {
    echo ""
    log_info "Botko Packing Service Installer"
    log_info "================================"
    echo ""

    check_prerequisites
    backup_existing_service
    install_service
    reload_systemd
    enable_service
    show_status
}

# Command routing
case "${1:-install}" in
    install)
        main_install
        ;;
    uninstall)
        check_prerequisites
        uninstall_service
        ;;
    reinstall)
        check_prerequisites
        uninstall_service
        echo ""
        main_install
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
