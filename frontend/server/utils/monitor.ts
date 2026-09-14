// 主日值守监控（mini 侧）: 定时探测 home 转录检查端口(:9091/status) + 本进程字幕流水,
// 主日窗口内开始/结束两个检查点异常时经 tellme webhook 通知管理员。
// 监控放 mini 的原因: 转录(home)本身就是被检对象——转录挂掉 = 检查端口拉取失败, 天然可检。
// 排程/防误报(宽限+二次确认)设计见 docs/2026-09-14_主日值守监控设计.md
import { pipelineStatus } from './pipelineStatus'

const DEFAULT_TELLME_WEBHOOK =
  'https://n8n.346751.xyz/webhook/51d38d7b-bc2c-4b81-81cc-158eb5e75c47'

const WEEKDAYS = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'] // getUTCDay 口径, 周日=0

interface TranscribeStatus {
  ok?: boolean
  uptime_sec?: number
  stream?: { state?: 'ok' | 'silent' | 'no_stream'; max_rms_db_30s?: number; last_data_at?: number | null }
  ws?: { recording?: boolean }
}

interface MonitorConfig {
  start: string
  end: string
  days: Set<number>
  startGraceMs: number
  endGraceMs: number
  confirmMs: number
  subsLookbackMs: number
  checkIntervalMs: number
  transcribeApi: string
  webhook: string
  dryRun: boolean
}

export function buildMonitorConfig(rc: Record<string, any>): MonitorConfig {
  const num = (v: any, def: number, min = 0) => {
    const n = Math.trunc(Number(v))
    return Number.isFinite(n) && n >= min ? n : def
  }
  const daysText = String(rc.monitorServiceDays || 'sun').trim().toLowerCase()
  const days = daysText === 'all' || daysText === 'daily' || daysText === '*'
    ? new Set(WEEKDAYS.map((_, i) => i))
    : new Set(WEEKDAYS.map((w, i) => (daysText.split(/[,\s]+/).includes(w) ? i : -1)).filter(i => i >= 0))
  return {
    start: String(rc.monitorServiceStart || '07:40'),
    end: String(rc.monitorServiceEnd || '09:25'),
    days: days.size ? days : new Set([0]), // 无法解析时缺省周日
    startGraceMs: num(rc.monitorStartGraceSec, 180) * 1000,
    endGraceMs: num(rc.monitorEndGraceSec, 120) * 1000,
    confirmMs: num(rc.monitorConfirmSec, 60) * 1000,
    subsLookbackMs: num(rc.monitorSubsLookbackSec, 1800) * 1000,
    checkIntervalMs: Math.max(5, num(rc.monitorCheckIntervalSec, 20)) * 1000,
    // 单一数据源: home 转录检查端口(run_client.py 暴露), 一次拉取含转录可达性+流三态;
    // 字幕取本进程流水(pipelineStatus), 不依赖其他服务
    transcribeApi: String(rc.monitorTranscribeApi || 'http://192.168.123.16:9091/status'),
    webhook: String(rc.tellmeWebhook || DEFAULT_TELLME_WEBHOOK),
    dryRun: rc.monitorDryRun === true || rc.monitorDryRun === 'true' || rc.monitorDryRun === '1',
  }
}

// 上海时间(显式 +8 偏移, 不依赖容器 TZ): { key: 'YYYY-MM-DD', hhmm: 'HH:MM', weekday: 0..6(周日=0) }
function shanghaiNow(now = Date.now()) {
  const d = new Date(now + 8 * 3_600_000)
  const p = (n: number) => String(n).padStart(2, '0')
  return {
    key: `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}`,
    hhmm: `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`,
    weekday: d.getUTCDay(),
  }
}

function parseHhmm(text: string): number | null {
  const m = /^(\d{1,2}):(\d{2})$/.exec(text.trim())
  if (!m) return null
  const h = Number(m[1]); const min = Number(m[2])
  return h >= 0 && h <= 23 && min >= 0 && min <= 59 ? h * 60 + min : null
}

// 今日窗口起点/终点的 epoch ms（Date.parse 带 +08:00 偏移, 不受容器 TZ 影响）
function windowAt(dateKey: string, hhmm: string): number {
  return Date.parse(`${dateKey}T${hhmm}:00+08:00`)
}

async function probeTranscribe(api: string): Promise<{ probe: TranscribeStatus | null; error: string | null }> {
  try {
    const r = await fetch(api, { signal: AbortSignal.timeout(5000) })
    if (!r.ok) return { probe: null, error: `HTTP ${r.status}` }
    return { probe: await r.json() as TranscribeStatus, error: null }
  } catch (e: any) {
    return { probe: null, error: String(e?.cause?.code || e?.name || 'Error') }
  }
}

// 值守是一次性通知: 任意环节出问题报告管理员后即由人跟进, 不做多源二次诊断——
// 因此转录不可达时不再另行探测 SRS/推流, 消息如实标注"无法探测"即可


async function notify(cfg: MonitorConfig, body: string, log: (s: string) => void): Promise<void> {
  const full = `[livetrans 值守] ${fmtFullShanghai()}\n${body}`
  if (cfg.dryRun) {
    log(`[DRY-RUN 通知]\n${full}`)
    return
  }
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const r = await fetch(cfg.webhook, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: full }),
        signal: AbortSignal.timeout(10000),
      })
      if (!r.ok) throw new Error(`webhook HTTP ${r.status}`)
      log('已通知管理员')
      return
    } catch (e: any) {
      log(`通知发送失败（第${attempt}次）: ${e?.message || e}`)
      if (attempt < 3) await new Promise(r => setTimeout(r, 3000))
    }
  }
  log('通知最终失败，消息丢弃')
}

function fmtFullShanghai(now = Date.now()): string {
  const d = new Date(now + 8 * 3_600_000)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ` +
    `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}`
}

function audioLine(probe: TranscribeStatus | null): string {
  if (!probe) return '❓ 未知（转录服务不可达，无法探测）'
  const s = probe.stream || {}
  const db = typeof s.max_rms_db_30s === 'number' ? s.max_rms_db_30s.toFixed(1) : '?'
  if (s.state === 'ok') return `✅ 正常（峰值 ${db} dB）`
  if (s.state === 'silent') return `⚠️ 有流但静音（峰值 ${db} dB）`
  return '❌ 无流'
}

function subsLine(lookbackMs: number): string {
  const subs = pipelineStatus()
  const fmt = (ms: number) => (ms ? shanghaiNow(ms).hhmm : '无')
  const look = `${Math.round(lookbackMs / 60000)} 分钟`
  if (subs.lastAt && subs.lastAt >= Date.now() - lookbackMs) {
    const conf = subs.lastConfirmedAt ? `，正式 ${fmt(subs.lastConfirmedAt)}` : ''
    return `✅ 已收到（最近 ${fmt(subs.lastAt)}${conf}）`
  }
  return `❌ 最近 ${look}未收到任何字幕（最近 ${fmt(subs.lastAt)}，正式 ${fmt(subs.lastConfirmedAt)}）`
}

export function startMonitor(rc: Record<string, any>): void {
  const cfg = buildMonitorConfig(rc)
  const log = (s: string) => console.log(`[MONITOR] ${s}`)
  const startMin = parseHhmm(cfg.start)
  const endMin = parseHhmm(cfg.end)
  if (startMin === null || endMin === null || endMin <= startMin) {
    log(`配置错误: 窗口 ${cfg.start}-${cfg.end} 非法（须同日且 END>START），监控停用`)
    return
  }
  const daysLabel = cfg.days.size === 7 ? '每天' : [...cfg.days].sort().map(d => WEEKDAYS[d]).join('/')
  log(`值守监控启动: ${daysLabel} ${cfg.start}-${cfg.end}（上海），宽限 开始+${cfg.startGraceMs / 1000}s/结束+${cfg.endGraceMs / 1000}s，` +
    `二次确认 ${cfg.confirmMs / 1000}s，字幕回看 ${cfg.subsLookbackMs / 60000}min，` +
    `转录=${cfg.transcribeApi}${cfg.dryRun ? '，DRY-RUN' : ''}`)

  // 当天检查状态（key=上海日期，跨天自然失效）
  const state = { startDoneKey: '', endDoneKey: '', pendingSince: 0 }

  async function startTick(now: number, dateKey: string, startAt: number): Promise<void> {
    const { probe, error } = await probeTranscribe(cfg.transcribeApi)
    const audioOk = !!probe && probe.stream?.state === 'ok'
    const subsHas = !!pipelineStatus().lastAt && pipelineStatus().lastAt >= now - cfg.subsLookbackMs
    const problem = !audioOk || !subsHas
    const transcribePart = probe
      ? `✅ 可达（uptime ${probe.uptime_sec ?? '?'}s）`
      : `❌ 不可达（${error}）`

    if (now < startAt + cfg.startGraceMs) {
      // 宽限期内只观察不通知: 等推流/开声就位
      log(`[开始检查] 宽限期内探测: 转录=${probe ? 'ok' : '不可达'} 音频=${probe?.stream?.state ?? '未知'} 字幕=${subsHas ? '有' : '无'}`)
      return
    }

    if (!problem) {
      if (state.pendingSince) log('[开始检查] 异常已恢复，不通知')
      state.pendingSince = 0
      state.startDoneKey = dateKey
      log(`[开始检查] 正常: 转录可达, 音频=${probe!.stream!.state}, 字幕=${subsHas ? '有' : '无'}`)
      return
    }

    if (!state.pendingSince) {
      state.pendingSince = now
      log(`[开始检查] 发现异常（转录=${probe ? 'ok' : `不可达(${error})`} 音频=${probe?.stream?.state ?? '未知'} 字幕=${subsHas ? '有' : '无'}），${cfg.confirmMs / 1000}s 后二次确认`)
      return
    }
    if (now - state.pendingSince < cfg.confirmMs) return

    const body =
      `⚠️ 主日礼拜已开始（${cfg.start}），检查异常且持续 ${cfg.confirmMs / 1000}s 未恢复：\n` +
      `· 转录服务 ${transcribePart}\n` +
      `· 音频源 ${audioLine(probe)}\n` +
      `· 字幕 ${subsLine(cfg.subsLookbackMs)}`
    await notify(cfg, body, log)
    state.pendingSince = 0
    state.startDoneKey = dateKey
  }

  async function endCheck(dateKey: string, endAt: number): Promise<void> {
    state.endDoneKey = dateKey
    const { probe, error } = await probeTranscribe(cfg.transcribeApi)
    if (probe) {
      const st = probe.stream?.state
      if (st === 'no_stream') {
        log('[结束检查] stream 已停播，无需通知')
        return
      }
      await notify(cfg,
        `⚠️ 已到结束时间（${cfg.end}），但 stream 仍在直播：\n` +
        `· 转录服务 ✅ 可达\n· 音频源 ${audioLine(probe)}\n请确认是否需要停止推流`, log)
      return
    }
    // 转录不可达=无法判断直播状态; 开始检查已报过"转录不可达", 管理员处理中, 不重复打扰
    log(`[结束检查] 转录服务不可达（${error}），无法确认直播状态，不重复通知`)
  }

  async function tick(): Promise<void> {
    try {
      const now = Date.now()
      const t = shanghaiNow(now)
      if (!cfg.days.has(t.weekday)) return
      const startAt = windowAt(t.key, cfg.start)
      const endAt = windowAt(t.key, cfg.end)

      if (state.startDoneKey !== t.key) {
        if (now >= endAt) {
          state.startDoneKey = t.key
          log('已过今日窗口，开始检查作废')
        } else if (now >= startAt) {
          await startTick(now, t.key, startAt)
        }
      }

      if (state.endDoneKey !== t.key && now >= endAt + cfg.endGraceMs) {
        await endCheck(t.key, endAt)
      }
    } catch (e: any) {
      log(`tick 异常: ${e?.stack || e}`)
    }
  }

  setInterval(tick, cfg.checkIntervalMs)
}
