#!/usr/bin/env bash
# health_check.sh — Check liveness of all LLM service endpoints.
#
# Install via crontab:
#   crontab -e
#   * * * * * /opt/monitoring/health_check.sh
#
# Logs are written to syslog (visible via `journalctl -t llm-health`) and to a
# rotating log file. The rotating file requires logrotate — see the config at
# the bottom of this script.
#
# Optional alerting: uncomment and configure the Slack webhook line below.

set -euo pipefail

# --- Configuration -----------------------------------------------------------

ENDPOINTS=(
    "http://10.x.x.10:8000/health"   # Spark #1: vLLM embedding
    "http://10.x.x.11:8000/health"   # Spark #2: vLLM chat
    "http://localhost:4000/health"    # LiteLLM gateway
    "http://localhost/health"         # nginx (if running on the gateway host)
)

LOG_FILE="/var/log/llm-health.log"
CURL_TIMEOUT=5   # seconds

# Uncomment to send Slack alerts:
# SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# -----------------------------------------------------------------------------

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" >> "$LOG_FILE"
    logger -t llm-health "$*"
}

alert() {
    local msg="$*"
    log "ALERT: $msg"

    # Slack alert (uncomment if SLACK_WEBHOOK_URL is set above):
    # if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
    #     curl -sf -X POST "$SLACK_WEBHOOK_URL" \
    #         -H "Content-Type: application/json" \
    #         -d "{\"text\": \":warning: LLM health check: $msg\"}" \
    #         > /dev/null 2>&1 || true
    # fi
}

for url in "${ENDPOINTS[@]}"; do
    if curl -sf --max-time "$CURL_TIMEOUT" "$url" > /dev/null 2>&1; then
        : # healthy — no log entry (keep the log quiet when everything is fine)
    else
        alert "UNHEALTHY: $url"
    fi
done

# --- Logrotate configuration --------------------------------------------------
# Create /etc/logrotate.d/llm-health with the following content to prevent the
# log file from growing unbounded:
#
# /var/log/llm-health.log {
#     daily
#     rotate 14
#     compress
#     delaycompress
#     missingok
#     notifempty
#     create 0640 root adm
# }
