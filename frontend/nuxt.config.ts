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
    // 观众报错反馈的 n8n webhook(tellme 同款通道), NUXT_TELLME_WEBHOOK 可运行时覆盖
    tellmeWebhook: process.env.TELLME_WEBHOOK,
  },
});
