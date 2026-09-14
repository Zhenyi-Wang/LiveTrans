// 值守监控数据源: 记录字幕链路最近流量的时间戳（进程内存态, 重启/部署归零）
// 监控(server/plugins/monitor.ts)同进程直接读取, 判断最近 N 分钟内是否收到过字幕;
// GET /backend/api/status 亦暴露同一份时间戳供人工排查。
// 只记"收到过", 不做回看窗口判断——窗口比较在监控侧做, 这里保持无状态口径。

// state 必须挂 globalThis: dev 模式下 plugin 与 route handler 可能各持一份模块副本,
// 模块级变量会分裂; globalThis 保证所有副本读写同一份(生产单 bundle 同样兼容)
const g = globalThis as any
const state: { lastAt: number; lastCurrentAt: number; lastConfirmedAt: number } =
  g.__livetransPipelineStatus ??= {
    lastAt: 0, // 任意 listen POST（含无效 segment）——链路通不通的口径
    lastCurrentAt: 0, // 实际广播过 current 预览
    lastConfirmedAt: 0, // 实际入队过 confirmed 正式字幕
  }

export function recordListen(kind: 'any' | 'current' | 'confirmed', at = Date.now()): void {
  state.lastAt = at
  if (kind === 'current') state.lastCurrentAt = at
  else if (kind === 'confirmed') state.lastConfirmedAt = at
}

export function pipelineStatus() {
  return { ...state }
}
