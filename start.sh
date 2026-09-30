#!/usr/bin/env bash
# 统一启动入口：幂等地在 tmux 会话中运行 ./run.sh
# 与 transcribe-service / asr-engine 的 start.sh 同源模板；本副本额外带 -e env 透传（livetrans 专属）
set -euo pipefail

SESSION="livetrans"
HEALTH_URL=""   # 就绪探测端点；留空跳过等待
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"

# 幂等：会话已存在则不重复创建
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux 会话 '$SESSION' 已在运行，未重复启动（tmux a -t $SESSION 查看）"
    if [[ -n "${LIVETRANS_NO_DATA_TIMEOUT:-}" ]]; then
        echo "⚠ 本次 LIVETRANS_NO_DATA_TIMEOUT=${LIVETRANS_NO_DATA_TIMEOUT} 未生效（会话沿用旧环境），需 tmux kill-session -t $SESSION 后重启"
    fi
    exit 0
fi

# -e 透传运行时可调参数: tmux server 常驻, new-session 默认不继承当前 shell 环境;
# 空值无害(client 端容错回落默认)。需要覆盖时: LIVETRANS_NO_DATA_TIMEOUT=15 ./start.sh
# 注意: -e 需 tmux >= 3.2 (本机 3.2a)
tmux new-session -d -s "$SESSION" -c "$DIR" \
    -e "LIVETRANS_NO_DATA_TIMEOUT=${LIVETRANS_NO_DATA_TIMEOUT:-}" \
    "./run.sh"
echo "✓ 已创建 tmux 会话 '$SESSION'，后台启动中..."

# 可选：等待健康端点就绪
if [[ -n "$HEALTH_URL" ]]; then
    echo "等待服务就绪（最多 25s）..."
    for _ in $(seq 1 25); do
        if curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
            echo "✓ 服务就绪: $HEALTH_URL"
            exit 0
        fi
        sleep 1
    done
    echo "⚠ 服务未在 25s 内就绪，查看输出: tmux a -t $SESSION"
    exit 1
fi
