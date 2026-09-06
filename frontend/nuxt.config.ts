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
    // 共享对话流保留的最大轮数(user/assistant一来一回为一轮)
    historyMaxRounds: Number(process.env.HISTORY_MAX_ROUNDS) || 20,
    // LLM单次调用超时(ms): 超时触发重试,防网关卡死占住并发槽拖垮串行翻译队列
    // 实测正常p99=8.5s,15s≈1.8×p99,正常抖动零误杀;可用 NUXT_LLM_TIMEOUT_MS 运行时覆盖
    // 钳位与使用点(ai.ts)同款: 非整数/负数同步抛RangeError,(2^31,2^32]被置1ms即超时,
    // 上限取setTimeout的32位有符号整数边界(运行时覆盖绕过此处钳位,ai.ts使用点是权威防线)
    llmTimeoutMs: Math.min(2147483647, Math.max(1000, Math.trunc(Number(process.env.LLM_TIMEOUT_MS)) || 15000)),
    // 观众报错反馈的 n8n webhook(tellme 同款通道), NUXT_TELLME_WEBHOOK 可运行时覆盖
    tellmeWebhook: process.env.TELLME_WEBHOOK,
  },
});
