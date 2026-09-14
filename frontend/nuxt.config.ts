// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: "2024-04-03",
  devtools: { enabled: true },
  modules: [
    "@pinia/nuxt",
    "@vueuse/nuxt",
    // '@element-plus/nuxt',
  ],
  app: {
    head: {
      title: "Live Translation | Haining Xiashi Christian Church",
      charset: "utf-8",
      viewport: "width=device-width, initial-scale=1",
      link: [
        { rel: 'icon', type: 'image/png', href: '/logo.jpg' }
      ]  
    },
  },
  nitro: {
    experimental: {
      websocket: true,
    },
  },
  runtimeConfig: {
    openaiApiKey: process.env.OPENAI_API_KEY,
    openaiBaseUrl: process.env.OPENAI_BASE_URL,
    openaiModel: process.env.OPENAI_MODEL,
    // confirmed批量翻译每批最大片段数,积压时按此大小循环补齐
    confirmedBatchMax: Number(process.env.CONFIRMED_BATCH_MAX) || 2,
    // 共享对话流窗口(锯齿式): 超过 MAX 轮一次裁到 MIN 轮,拉长前缀断裂间隔以保 DeepSeek 缓存命中率
    historyMinRounds: Number(process.env.HISTORY_MIN_ROUNDS) || 20,
    historyMaxRounds: Number(process.env.HISTORY_MAX_ROUNDS) || 100,
    // current 预览 LLM 调用最小间隔(ms,start-to-start),限流中间版本翻译; NUXT_PREVIEW_MIN_INTERVAL_MS 可运行时覆盖
    previewMinIntervalMs: Number(process.env.PREVIEW_MIN_INTERVAL_MS) || 3000,
    // LLM单次调用超时(ms): 超时触发重试,防网关卡死占住并发槽拖垮串行翻译队列
    // 实测正常p99=8.5s,15s≈1.8×p99,正常抖动零误杀;可用 NUXT_LLM_TIMEOUT_MS 运行时覆盖
    // 钳位与使用点(ai.ts)同款: 非整数/负数同步抛RangeError,(2^31,2^32]被置1ms即超时,
    // 上限取setTimeout的32位有符号整数边界(运行时覆盖绕过此处钳位,ai.ts使用点是权威防线)
    llmTimeoutMs: Math.min(2147483647, Math.max(1000, Math.trunc(Number(process.env.LLM_TIMEOUT_MS)) || 15000)),
    // 观众报错反馈的 n8n webhook(tellme 同款通道), NUXT_TELLME_WEBHOOK 可运行时覆盖
    tellmeWebhook: process.env.TELLME_WEBHOOK,
    // ===== 主日值守监控(mini 侧, server/plugins/monitor.ts) =====
    // 默认主日(周日) 07:40-09:25: 开始检查转录/音频/字幕三状态, 结束检查是否仍在直播
    monitorEnabled: process.env.MONITOR_ENABLED !== "0",
    monitorServiceStart: process.env.MONITOR_SERVICE_START || "07:40",
    monitorServiceEnd: process.env.MONITOR_SERVICE_END || "09:25",
    monitorServiceDays: process.env.MONITOR_SERVICE_DAYS || "sun",
    monitorStartGraceSec: process.env.MONITOR_START_GRACE_SEC || 180,
    monitorEndGraceSec: process.env.MONITOR_END_GRACE_SEC || 120,
    monitorConfirmSec: process.env.MONITOR_CONFIRM_SEC || 60,
    // 字幕"有/无"判定滚动回看窗口(秒)
    monitorSubsLookbackSec: process.env.MONITOR_SUBS_LOOKBACK_SEC || 1800,
    monitorCheckIntervalSec: process.env.MONITOR_CHECK_INTERVAL_SEC || 20,
    // home 转录检查端口(run_client.py 暴露), 单一数据源一次拉全
    monitorTranscribeApi: process.env.MONITOR_TRANSCRIBE_API || "http://192.168.123.16:9091/status",
    // 非空字符串 "1"/"true" 时只打印不发送通知
    monitorDryRun: process.env.MONITOR_DRY_RUN || false,
  },
});
