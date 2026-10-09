#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
CUDA_HOST="${CUDA_HOST:-cuda}"
A_HTTP_PORT="${A_HTTP_PORT:-17900}"
C_HTTP_PORT="${C_HTTP_PORT:-17891}"
LOG_DIR="${LOG_DIR:-$HOME/Library/Logs}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/trae-agent-proxy}"
GOST_PID_FILE="${GOST_PID_FILE:-$HOME/.trae-gost.pid}"
GOST_LOG="$LOG_DIR/trae-gost.log"
SSH_LOG="$LOG_DIR/trae-proxy-ssh.log"
SSH_CONTROL="$STATE_DIR/ssh-control.sock"

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

validate_port() {
    case "$1" in
        ''|*[!0-9]*) die "invalid port: $1" ;;
    esac

    if [ "$1" -lt 1 ] || [ "$1" -gt 65535 ]; then
        die "port out of range: $1"
    fi
}

gost_pid() {
    if [ -f "$GOST_PID_FILE" ]; then
        cat "$GOST_PID_FILE"
    fi
}

gost_is_running() {
    local pid
    pid="$(gost_pid)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

local_http_proxy_is_listening() {
    lsof -nP -iTCP:"$A_HTTP_PORT" -sTCP:LISTEN 2>/dev/null |
        grep -q "127.0.0.1:$A_HTTP_PORT"
}

ssh_master_is_running() {
    [ -S "$SSH_CONTROL" ] &&
        ssh -S "$SSH_CONTROL" -O check "$CUDA_HOST" >/dev/null 2>&1
}

remote_http_proxy_is_listening() {
    ssh -S "$SSH_CONTROL" -o ClearAllForwardings=yes "$CUDA_HOST" \
        "ss -ltn | grep -q '127.0.0.1:${C_HTTP_PORT}'"
}

start_gost() {
    local pid

    if gost_is_running; then
        log "gost_status=already_running"
        return
    fi

    if local_http_proxy_is_listening; then
        die "127.0.0.1:$A_HTTP_PORT is already used by an unmanaged process"
    fi

    rm -f "$GOST_PID_FILE"
    nohup gost -L "http://127.0.0.1:$A_HTTP_PORT" \
        >"$GOST_LOG" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" >"$GOST_PID_FILE"

    sleep 1
    if ! kill -0 "$pid" 2>/dev/null || ! local_http_proxy_is_listening; then
        tail -n 40 "$GOST_LOG" >&2 || true
        die "gost failed to listen on 127.0.0.1:$A_HTTP_PORT"
    fi

    log "gost_status=started"
    log "gost_pid=$pid"
}

start_ssh_tunnel() {
    local attempt=0

    if ssh_master_is_running; then
        log "ssh_tunnel_status=already_running"
        return
    fi

    rm -f "$SSH_CONTROL"
    ssh -M -S "$SSH_CONTROL" -fN \
        -o ControlPersist=no \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o "LogLevel=ERROR" \
        -E "$SSH_LOG" \
        -R "127.0.0.1:$C_HTTP_PORT:127.0.0.1:$A_HTTP_PORT" \
        "$CUDA_HOST"

    while [ "$attempt" -lt 20 ]; do
        if ssh_master_is_running && remote_http_proxy_is_listening; then
            log "ssh_tunnel_status=started"
            return
        fi
        attempt=$((attempt + 1))
        sleep 0.25
    done

    tail -n 40 "$SSH_LOG" >&2 || true
    die "SSH tunnel did not expose C port $C_HTTP_PORT"
}

verify_proxy() {
    local a_ip
    local c_ip
    local api_code

    a_ip="$(
        curl -4 -fsS --max-time 15 \
            --proxy "http://127.0.0.1:$A_HTTP_PORT" \
            https://api.ipify.org
    )"
    c_ip="$(
        ssh -S "$SSH_CONTROL" -o ClearAllForwardings=yes "$CUDA_HOST" \
            "curl -4 -fsS --max-time 15 \
                --proxy http://127.0.0.1:$C_HTTP_PORT \
                https://api.ipify.org"
    )"
    api_code="$(
        ssh -S "$SSH_CONTROL" -o ClearAllForwardings=yes "$CUDA_HOST" \
            "curl -4 -sS -o /dev/null -w '%{http_code}' --max-time 15 \
                --proxy http://127.0.0.1:$C_HTTP_PORT \
                https://api.trae.com.cn/"
    )"

    log "a_proxy_egress_ip=$a_ip"
    log "c_proxy_egress_ip=$c_ip"
    log "trae_api_root_http_code=$api_code"

    if [ "$api_code" != "404" ]; then
        die "expected Trae API root to return HTTP 404"
    fi
}

status() {
    local failed=0

    if gost_is_running && local_http_proxy_is_listening; then
        log "gost_status=running"
        log "gost_pid=$(gost_pid)"
    else
        log "gost_status=stopped"
        failed=1
    fi

    if ssh_master_is_running; then
        log "ssh_tunnel_status=running"
        if remote_http_proxy_is_listening; then
            log "remote_listener=127.0.0.1:$C_HTTP_PORT"
        else
            log "remote_listener=missing"
            failed=1
        fi
    else
        log "ssh_tunnel_status=stopped"
        failed=1
    fi

    return "$failed"
}

stop_ssh_tunnel() {
    if ssh_master_is_running; then
        ssh -S "$SSH_CONTROL" -O exit "$CUDA_HOST" >/dev/null
        log "ssh_tunnel_status=stopped"
    else
        log "ssh_tunnel_status=already_stopped"
    fi
    rm -f "$SSH_CONTROL"
}

stop_gost() {
    local attempt=0
    local pid

    if ! gost_is_running; then
        log "gost_status=already_stopped"
        rm -f "$GOST_PID_FILE"
        return
    fi

    pid="$(gost_pid)"
    kill -TERM "$pid"
    while [ "$attempt" -lt 20 ] && kill -0 "$pid" 2>/dev/null; do
        attempt=$((attempt + 1))
        sleep 0.25
    done

    if kill -0 "$pid" 2>/dev/null; then
        die "gost process $pid did not stop"
    fi

    rm -f "$GOST_PID_FILE"
    log "gost_status=stopped"
}

start() {
    start_gost
    start_ssh_tunnel
    verify_proxy
}

require_command gost
require_command ssh
require_command curl
require_command lsof
validate_port "$A_HTTP_PORT"
validate_port "$C_HTTP_PORT"
mkdir -p "$LOG_DIR" "$STATE_DIR"

case "$ACTION" in
    start)
        start
        ;;
    verify)
        status
        verify_proxy
        ;;
    status)
        status
        ;;
    stop)
        stop_ssh_tunnel
        stop_gost
        ;;
    restart)
        stop_ssh_tunnel
        stop_gost
        start
        ;;
    *)
        die "usage: $0 {start|verify|status|stop|restart}"
        ;;
esac
