# Systemd Service Setup for Botko Packing

## Overview

This directory contains systemd service files and management scripts for running the Botko 3D bin packing experiments as a managed background service.

## Files Created

1. **botko-packing.service** - Systemd unit file
2. **service_manager.sh** - Service management wrapper script
3. **install_service.sh** - Automated installation script

## Quick Start

### Prerequisites

1. Ensure virtual environment is created:
   ```bash
   cd /home/louis/Box-Simulations-botko
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Ensure the experiment runner module exists:
   ```bash
   # Should exist: src/runner/experiment.py
   ```

### Installation

```bash
# Install and enable the service
cd /home/louis/Box-Simulations-botko
./scripts/install_service.sh
```

This will:
- Copy service file to /etc/systemd/system/
- Set correct permissions (644)
- Reload systemd daemon
- Enable service for boot startup

### Service Management

Use the service manager script for all operations:

```bash
# Start the service
./scripts/service_manager.sh start

# Stop the service
./scripts/service_manager.sh stop

# Restart the service
./scripts/service_manager.sh restart

# Check status
./scripts/service_manager.sh status

# View logs (last 50 lines)
./scripts/service_manager.sh logs

# View logs (last 100 lines)
./scripts/service_manager.sh logs 100

# Follow logs in real-time
./scripts/service_manager.sh logs follow

# Enable auto-start on boot
./scripts/service_manager.sh enable

# Disable auto-start on boot
./scripts/service_manager.sh disable
```

## Service Configuration

The service is configured with the following settings:

- **User**: louis
- **Working Directory**: /home/louis/Box-Simulations-botko
- **Entry Point**: `python -m src.runner.experiment`
- **CPU Quota**: 50% (limits CPU usage)
- **Nice Level**: 10 (lower priority)
- **Restart Policy**: on-failure (automatic restart on crashes)
- **Restart Delay**: 30 seconds
- **Logging**: systemd journal (journalctl)

### CPU and Priority Settings

- **CPUQuota=50%**: Limits the service to 50% of one CPU core
- **Nice=10**: Runs with lower priority, won't interfere with interactive tasks
- Both settings ensure experiments run overnight without impacting system responsiveness

## Safety Features

### Service Manager Checks

The service_manager.sh script includes several safety checks:

1. **Virtual Environment Validation**: Ensures venv exists before starting
2. **Service Installation Check**: Verifies service is installed before operations
3. **Status Reporting**: Shows service state after start/restart operations
4. **Color-coded Output**: Green (info), yellow (warnings), red (errors)

### Installation Script Features

1. **Prerequisites Check**: Validates venv and required files
2. **Automatic Backup**: Backs up existing service file before overwrite
3. **Graceful Stop**: Stops running service before reinstallation
4. **Uninstall Support**: Clean removal with `./scripts/install_service.sh uninstall`

## Troubleshooting

### Service Won't Start

1. Check virtualenv:
   ```bash
   ls -la /home/louis/Box-Simulations-botko/venv/bin/python
   ```

2. Check experiment module:
   ```bash
   source venv/bin/activate
   python -c "import src.runner.experiment"
   ```

3. Check service logs:
   ```bash
   ./scripts/service_manager.sh logs 100
   ```

### High CPU Usage

The service is configured with CPUQuota=50%, but you can adjust this:

1. Edit scripts/botko-packing.service
2. Change `CPUQuota=50%` to desired percentage (e.g., `CPUQuota=25%`)
3. Reinstall: `./scripts/install_service.sh reinstall`

### View Real-time Logs

```bash
# Follow logs (Ctrl+C to exit)
./scripts/service_manager.sh logs follow

# Or use journalctl directly
journalctl -u botko-packing -f
```

## Uninstallation

```bash
# Stop and remove service
./scripts/install_service.sh uninstall
```

This will:
- Stop the running service
- Disable auto-start
- Remove service file from /etc/systemd/system/
- Reload systemd daemon

## Manual systemctl Commands

If needed, you can use systemctl directly:

```bash
# Start
sudo systemctl start botko-packing

# Stop
sudo systemctl stop botko-packing

# Restart
sudo systemctl restart botko-packing

# Status
systemctl status botko-packing

# Enable on boot
sudo systemctl enable botko-packing

# Disable on boot
sudo systemctl disable botko-packing

# View logs
journalctl -u botko-packing -n 50
```

## Testing Before Installation

Before installing the service, test manually:

```bash
cd /home/louis/Box-Simulations-botko
source venv/bin/activate
python -m src.runner.experiment
```

Ensure the experiment runs without errors before installing as a service.

## Notes

- **NOT installed yet**: Files are created but service is not installed to systemd
- **Test manually first**: Verify experiment.py works before installing service
- **Logs via journalctl**: All output goes to systemd journal
- **Auto-restart**: Service restarts automatically on failure (30s delay)
- **Boot persistence**: Service starts automatically on system boot (when enabled)
