# livetrans"一直重启、Ctrl+C 停不下"排查与修复

日期：2026-09-17（周三下午排查）

## 现象

- livetrans 在窗格里反复重启（`[MAIN] Client crashed ... restarting`）；
- Ctrl+C 按了停不下来；
- 用户以为是 systemd 反复拉起（实际不是）。

## 三个相互独立的问题叠加

### 1. "一直重启" = main.py 监督循环按设计无限拉起

`main.py` 是监督进程：client（`whisperlive/run_client.py`）一退出就以 5s→10s→…→60s 封顶的指数退避**无限重启**，没有尝试上限。client 持续崩溃时窗格就是无限的 restart 刷屏。client 当天反复崩的直接触发源已不可考（证据随 tmux 段错误丢失），可疑窗口：当天上午另一会话在改前端并往 mini `--force-recreate` 部署，部署窗口内直播流(:8080)/listen API(:8081) 短暂中断。

### 2. "Ctrl+C 停不下" = 旧信号处理只设标志位、永远等 client

旧代码 SIGINT/SIGTERM handler 只置 `exit_requested=True` 然后"waiting for client to exit"，主循环阻塞在 `process.wait()`。client 若卡死在不可中断调用（实测 CUDA 卸载会进入 D 状态，见下），永远等不到，也没有强杀路径。

实测佐证：client 强杀后，`run_server.py` 收 SIGTERM 关闭监听端口后仍会在 **D 状态**滞留数秒（CUDA teardown）才消失——GPU 驱动调用确实存在不可中断窗口。

### 3. 压垮一切：tmux 服务器段错误（14:11:29）

`tmux: server` 空指针段错误（SIGSEGV, tmux 3.2a @ WSL2 内核日志确认），**整个 tmux 服务器崩溃**，所有会话（livetrans/transcribe/asr/w/w-livetrans/w-mryk）连同进程全灭。三个 systemd 用户单元都是 `Type=oneshot + RemainAfterExit`，只负责开机拉起、不会自动重拉，之后单元仍显示 active (exited) 但实际全死。

## 修复内容（2026-09-17 已生效）

`main.py`：
- 信号 handler 重写：**第一次** Ctrl+C 先给 client 5s 自行优雅退出（它同样收到 SIGINT，finally 会写 SRT + stop_server），5s 后 terminate、12s 后 kill 兜底；**连续第二次** Ctrl+C 立即 terminate+2s kill（用户不耐烦语义）
- 退出前兜底 `_kill_port_users(9090)`（复用 client 的 lsof 逻辑）：client 被强杀时 server 变孤儿占端口/显存，由监督进程清场
- 实测：单次 Ctrl+C 数秒内会话退出、9090/9091 释放、GPU 释放、无残留

`run.sh`：`pkill -f "python main.py"` 误匹配 docker 容器（mryk24）root 进程报 Operation not permitted，改为 `pkill -u "$USER" ... || true`

## 附带发现与警示

- **tmux 服务器住在 livetrans.service 的 cgroup 里**：开机第一个 `start.sh` 的 tmux 进程成为服务器，之后所有会话（含其他服务、交互会话）都挂在它的 cgroup 下。**`systemctl --user stop livetrans` 会杀掉整个 tmux 服务器（所有会话陪葬）**，停止 livetrans 请用 `tmux kill-session -t livetrans`。
- 事故后恢复三个服务：单元虽显示 active 但进程全灭，直接 `systemctl --user restart asr-engine transcribe` + `./start.sh` 即可（start.sh 幂等）。
- tmux 3.2a（Ubuntu 22.04 apt 版）段错误根因未查，当日仅一次；若复发考虑升级 tmux。

## 相关

- 架构与监督设计见项目根 CLAUDE.md「完整服务启动」
- client 侧断流看护（停 server 释放 GPU、90min 退避）：`whisper_live/client.py` STREAM 断流处理段

## 后记（同日）：自启机制整体改造

事故暴露的 systemd 单元结构性问题（tmux 服务器宿主随机、stop/restart 语义错乱、oneshot 状态失真）同日按用户决策整体下线：三个用户单元停用并备份至 `~/.config/systemd/user/disabled-2026-09-17/`，改为 `~/.bashrc` 末尾自启块——按 boot_id 每开机周期仅首个交互 shell 拉起一次（以 boot_id 命名的标记目录 + mkdir 原子性，多 shell 并发安全），启动顺序 asr-engine → transcribe → livetrans。代价与注意：**开机后无人登录则服务不启动**（主日直播前若 home 重启过，mini 值守监控会发现转录不可达并 tellme 提醒，届时 SSH 登录一次即自动拉起）。曾考虑并放弃的替代方案：per-service `tmux -L` 独立 socket（语义最优但要改三台机器 attach 习惯）、master 服务托管共享 tmux 服务器（不解决单点段错误全灭）、`/etc/wsl.conf [boot] command`（真正开机即起，以 root 运行，备选）。
