import { startMonitor } from '../utils/monitor'

// 主日值守监控启动入口: Nitro 进程内定时器, 与前端服务同生命周期(容器常驻)
// MONITOR_ENABLED=0 关闭; globalThis 守卫防 dev 模式 HMR 重复拉起多个监控实例
export default defineNitroPlugin(() => {
  const rc = useRuntimeConfig()
  if (rc.monitorEnabled === false || rc.monitorEnabled === 'false' || rc.monitorEnabled === '0') {
    console.log('[MONITOR] MONITOR_ENABLED=0，值守监控未启动')
    return
  }
  const g = globalThis as any
  if (g.__livetransMonitorStarted) return
  g.__livetransMonitorStarted = true
  startMonitor(rc)
})
