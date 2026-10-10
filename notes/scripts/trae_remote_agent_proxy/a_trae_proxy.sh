#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOST="${CUDA_HOST:-cuda}"
A_HTTP_PORT="${A_HTTP_PORT:-17900}"
C_HTTP_PORT="${C_HTTP_PORT:-17891}"
A_SOURCE_IP="${A_SOURCE_IP:-}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/trae-agent-proxy}"
GOST_PID_FILE="${GOST_PID_FILE:-$HOME/.trae-gost.pid}"
C_HELPER_LOCAL="${C_HELPER_LOCAL:-$SCRIPT_DIR/c_configure_trae_proxy.sh}"
C_HELPER_REMOTE_REL="${C_HELPER_REMOTE_REL:-.local/bin/c_configure_trae_proxy.sh}"
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

deploy_c_helper() {
    local remote_temp="$C_HELPER_REMOTE_REL.upload.$$"

    [ -f "$C_HELPER_LOCAL" ] ||
        die "C helper not found next to A script: $C_HELPER_LOCAL"

    ssh -o ClearAllForwardings=yes "$CUDA_HOST" \
        'mkdir -p "$HOME/.local/bin"'
    scp -q -o ClearAllForwardings=yes \
        "$C_HELPER_LOCAL" "$CUDA_HOST:$remote_temp"
    ssh -o ClearAllForwardings=yes "$CUDA_HOST" \
        "chmod 0755 '$remote_temp' &&
         mv -f '$remote_temp' '$C_HELPER_REMOTE_REL'"

    log "c_helper_status=deployed"
}

resolve_a_source_ip() {
    local source_ip="$A_SOURCE_IP"

    if [ -z "$source_ip" ]; then
        source_ip="$(
            ssh -o ClearAllForwardings=yes "$CUDA_HOST" \
                'printf "%s\n" "${SSH_CLIENT%% *}"'
        )"
    fi

    case "$source_ip" in
        ''|*[!0-9a-fA-F:.]*)
            die "invalid A source IP observed by C: $source_ip"
            ;;
    esac

    printf '%s\n' "$source_ip"
}

run_c_helper() {
    local action="$1"
    local source_ip

    source_ip="$(resolve_a_source_ip)"
    ssh -o ClearAllForwardings=yes "$CUDA_HOST" \
        "export A_SOURCE_IP='$source_ip'
         export C_HTTP_PORT='$C_HTTP_PORT'
         \"\$HOME/$C_HELPER_REMOTE_REL\" '$action'"
}

gost_pid() {
    if [ -f "$GOST_PID_FILE" ]; then
        cat "$GOST_PID_FILE"
    fi
}

gost_is_running() {
    local pid
    local listener_pid

    pid="$(gost_pid)"
    listener_pid="$(local_http_proxy_listener_pid)"
    [ -n "$pid" ] &&
        [ "$pid" = "$listener_pid" ] &&
        kill -0 "$pid" 2>/dev/null
}

local_http_proxy_listener_pid() {
    lsof -nP -t \
        -iTCP@127.0.0.1:"$A_HTTP_PORT" \
        -sTCP:LISTEN 2>/dev/null |
        head -n 1 ||
        true
}

local_http_proxy_is_listening() {
    [ -n "$(local_http_proxy_listener_pid)" ]
}

ssh_master_is_running() {
    [ -S "$SSH_CONTROL" ] &&
        ssh -S "$SSH_CONTROL" -O check "$CUDA_HOST" >/dev/null 2>&1
}

remote_http_proxy_is_listening() {
    ssh -S "$SSH_CONTROL" -o ClearAllForwardings=yes "$CUDA_HOST" \
        "ss -ltn | grep -q '127.0.0.1:${C_HTTP_PORT}'"
}

remote_http_proxy_port_is_in_use() {
    ssh -o ClearAllForwardings=yes "$CUDA_HOST" \
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
        >/dev/null 2>&1 &
    pid=$!
    printf '%s\n' "$pid" >"$GOST_PID_FILE"

    sleep 1
    if ! gost_is_running; then
        kill -TERM "$pid" 2>/dev/null || true
        rm -f "$GOST_PID_FILE"
        die "gost failed to listen on 127.0.0.1:$A_HTTP_PORT"
    fi

    log "gost_status=started"
    log "gost_pid=$pid"
}

start_ssh_tunnel() {
    local attempt=0

    if ssh_master_is_running; then
        if remote_http_proxy_is_listening; then
            log "ssh_tunnel_status=already_running"
            return
        fi

        ssh -S "$SSH_CONTROL" -O exit "$CUDA_HOST" >/dev/null || true
        rm -f "$SSH_CONTROL"
    fi

    if remote_http_proxy_port_is_in_use; then
        die "C port $C_HTTP_PORT is occupied by an unmanaged tunnel; stop the old manual ssh -R process on A once"
    fi

    rm -f "$SSH_CONTROL"
    ssh -M -S "$SSH_CONTROL" -fN \
        -o ControlPersist=no \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o "LogLevel=ERROR" \
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
    local listener_pid
    local pid

    if ! gost_is_running; then
        listener_pid="$(local_http_proxy_listener_pid)"
        if [ -n "$listener_pid" ]; then
            die "127.0.0.1:$A_HTTP_PORT is owned by unmanaged PID $listener_pid"
        fi

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

start_all() {
    start_gost
    start_ssh_tunnel
    verify_proxy
    run_c_helper enable
}

stop_all() {
    local failed=0

    if ! run_c_helper disable; then
        log "c_disable_status=failed"
        failed=1
    fi

    stop_ssh_tunnel
    stop_gost
    return "$failed"
}

restart_all() {
    stop_ssh_tunnel
    stop_gost
    start_all
}

status_all() {
    local failed=0

    status || failed=1
    run_c_helper status || failed=1
    return "$failed"
}

verify_all() {
    local failed=0

    status || failed=1
    verify_proxy || failed=1
    run_c_helper verify || failed=1
    return "$failed"
}

require_command gost
require_command ssh
require_command scp
require_command curl
require_command lsof
validate_port "$A_HTTP_PORT"
validate_port "$C_HTTP_PORT"
mkdir -p "$STATE_DIR"
deploy_c_helper

case "$ACTION" in
    start)
        start_all
        ;;
    verify)
        verify_all
        ;;
    status)
        status_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        restart_all
        ;;
    *)
        die "usage: $0 {start|verify|status|stop|restart}"
        ;;
esac
