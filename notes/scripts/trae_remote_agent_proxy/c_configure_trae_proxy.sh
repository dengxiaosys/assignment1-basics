#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
A_SOURCE_IP="${A_SOURCE_IP:-192.168.71.121}"
C_HTTP_PORT="${C_HTTP_PORT:-17891}"
PROFILE_PATH="${PROFILE_PATH:-$HOME/.profile}"
NO_PROXY_VALUE="${NO_PROXY_VALUE:-127.0.0.1,localhost,::1,192.168.71.0/24}"
START_MARKER="# >>> trae-remote-agent-proxy >>>"
END_MARKER="# <<< trae-remote-agent-proxy <<<"
PROXY_URL="http://127.0.0.1:$C_HTTP_PORT"
TEMP_PROFILE=""

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'error=%s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

cleanup_temp_profile() {
    if [ -n "$TEMP_PROFILE" ]; then
        rm -f "$TEMP_PROFILE"
    fi
}

validate_inputs() {
    case "$A_SOURCE_IP" in
        ''|*[!0-9a-fA-F:.]*) die "invalid A_SOURCE_IP: $A_SOURCE_IP" ;;
    esac

    case "$C_HTTP_PORT" in
        ''|*[!0-9]*) die "invalid C_HTTP_PORT: $C_HTTP_PORT" ;;
    esac

    if [ "$C_HTTP_PORT" -lt 1 ] || [ "$C_HTTP_PORT" -gt 65535 ]; then
        die "C_HTTP_PORT out of range: $C_HTTP_PORT"
    fi
}

proxy_is_listening() {
    ss -ltn 2>/dev/null |
        grep -q "127.0.0.1:$C_HTTP_PORT"
}

verify_proxy() {
    local api_code
    local egress_ip

    proxy_is_listening ||
        die "HTTP proxy is not listening on 127.0.0.1:$C_HTTP_PORT"

    egress_ip="$(
        curl -4 -fsS --max-time 15 \
            --proxy "$PROXY_URL" \
            https://api.ipify.org
    )"
    api_code="$(
        curl -4 -sS -o /dev/null -w '%{http_code}' --max-time 15 \
            --proxy "$PROXY_URL" \
            https://api.trae.com.cn/
    )"

    log "proxy_egress_ip=$egress_ip"
    log "trae_api_root_http_code=$api_code"

    if [ "$api_code" != "404" ]; then
        die "expected Trae API root to return HTTP 404"
    fi
}

validate_marker_state() {
    local start_count
    local end_count

    start_count="$(grep -Fxc "$START_MARKER" "$PROFILE_PATH" 2>/dev/null || true)"
    end_count="$(grep -Fxc "$END_MARKER" "$PROFILE_PATH" 2>/dev/null || true)"

    if [ "$start_count" -gt 1 ] || [ "$end_count" -gt 1 ]; then
        die "duplicate Trae proxy markers in $PROFILE_PATH"
    fi

    if [ "$start_count" -ne "$end_count" ]; then
        die "unbalanced Trae proxy markers in $PROFILE_PATH"
    fi
}

write_profile_without_proxy_block() {
    local output_path="$1"

    awk -v start="$START_MARKER" -v end="$END_MARKER" '
        $0 == start {
            inside = 1
            next
        }
        $0 == end {
            inside = 0
            next
        }
        !inside {
            print
        }
    ' "$PROFILE_PATH" >"$output_path"
}

append_proxy_block() {
    local output_path="$1"

    cat >>"$output_path" <<EOF

$START_MARKER
# Route Trae Remote agent traffic through the HTTP tunnel on Mac A.
case "\${SSH_CLIENT:-}" in
    $A_SOURCE_IP\\ *)
        _trae_proxy="$PROXY_URL"

        export HTTP_PROXY="\$_trae_proxy"
        export HTTPS_PROXY="\$_trae_proxy"
        export ALL_PROXY="\$_trae_proxy"
        export http_proxy="\$_trae_proxy"
        export https_proxy="\$_trae_proxy"
        export all_proxy="\$_trae_proxy"

        export NO_PROXY="$NO_PROXY_VALUE"
        export no_proxy="\$NO_PROXY"
        unset _trae_proxy
        ;;
esac
$END_MARKER
EOF
}

install_profile() {
    local new_profile="$1"
    local backup

    bash -n "$new_profile"
    backup="$PROFILE_PATH.backup-before-trae-proxy-$(date +%Y%m%d-%H%M%S-%N)"
    cp -p "$PROFILE_PATH" "$backup"
    chmod --reference="$PROFILE_PATH" "$new_profile"
    mv "$new_profile" "$PROFILE_PATH"
    log "profile_backup=$backup"
}

apply_config() {
    verify_proxy
    touch "$PROFILE_PATH"
    validate_marker_state
    TEMP_PROFILE="$(mktemp "$HOME/.profile.trae-proxy.XXXXXX")"

    write_profile_without_proxy_block "$TEMP_PROFILE"
    append_proxy_block "$TEMP_PROFILE"
    install_profile "$TEMP_PROFILE"
    TEMP_PROFILE=""

    log "profile_status=configured"
    log "proxy_url=$PROXY_URL"
    log "a_source_ip=$A_SOURCE_IP"
}

remove_config() {
    touch "$PROFILE_PATH"
    validate_marker_state
    if ! grep -Fqx "$START_MARKER" "$PROFILE_PATH"; then
        log "profile_status=already_unconfigured"
        return
    fi

    TEMP_PROFILE="$(mktemp "$HOME/.profile.trae-proxy.XXXXXX")"
    write_profile_without_proxy_block "$TEMP_PROFILE"
    install_profile "$TEMP_PROFILE"
    TEMP_PROFILE=""

    log "profile_status=unconfigured"
}

manager_pids() {
    pgrep -f '/index_trae[.]js --start-server' || true
}

restart_trae() {
    local current_pgid
    local pid
    local pgid
    local seen=" "
    local found=0

    current_pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"

    for pid in $(manager_pids); do
        pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
        case "$seen" in
            *" $pgid "*) continue ;;
        esac

        if [ "$pgid" = "$current_pgid" ]; then
            die "run restart-trae from an independent SSH session, not a Trae terminal"
        fi

        seen="$seen$pgid "
        found=1
        kill -TERM -- "-$pgid"
        log "terminated_trae_process_group=$pgid"
    done

    if [ "$found" -eq 0 ]; then
        log "trae_status=not_running"
    else
        log "trae_status=restart_requested"
    fi
}

profile_is_configured() {
    grep -Fqx "$START_MARKER" "$PROFILE_PATH" 2>/dev/null &&
        grep -Fqx "$END_MARKER" "$PROFILE_PATH" 2>/dev/null &&
        grep -Fq "_trae_proxy=\"$PROXY_URL\"" "$PROFILE_PATH"
}

status() {
    local failed=0
    local pid

    if proxy_is_listening; then
        log "proxy_listener=127.0.0.1:$C_HTTP_PORT"
    else
        log "proxy_listener=missing"
        failed=1
    fi

    if profile_is_configured; then
        log "profile_status=configured"
    else
        log "profile_status=not_configured"
        failed=1
    fi

    pid="$(pgrep -f '/modules/ai-agent/ai-agent$' | head -n 1 || true)"
    if [ -n "$pid" ]; then
        log "ai_agent_pid=$pid"
        if tr '\0' '\n' <"/proc/$pid/environ" |
            grep -Fqx "HTTP_PROXY=$PROXY_URL"; then
            log "ai_agent_proxy_status=active"
        else
            log "ai_agent_proxy_status=stale_or_missing"
            failed=1
        fi
    else
        log "ai_agent_pid=not_running"
        failed=1
    fi

    return "$failed"
}

require_command awk
require_command curl
require_command grep
require_command pgrep
require_command ps
require_command ss
validate_inputs
trap cleanup_temp_profile EXIT

case "$ACTION" in
    apply)
        apply_config
        ;;
    apply-and-restart)
        apply_config
        restart_trae
        ;;
    verify)
        verify_proxy
        status
        ;;
    status)
        status
        ;;
    restart-trae)
        restart_trae
        ;;
    remove)
        remove_config
        ;;
    remove-and-restart)
        remove_config
        restart_trae
        ;;
    *)
        die "usage: $0 {apply|apply-and-restart|verify|status|restart-trae|remove|remove-and-restart}"
        ;;
esac
