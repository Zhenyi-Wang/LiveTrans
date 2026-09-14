import { pipelineStatus } from '../../../utils/pipelineStatus'

// 字幕链路最近流量时间戳（只读, 无敏感信息）: 供人工排查浏览器直看,
// 监控(server/plugins/monitor.ts)同进程直接读 pipelineStatus(), 不经此端点
export default defineEventHandler(() => ({
  ok: true,
  now: Date.now(),
  listen: pipelineStatus(),
}))
