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
      title: "Live Translation | Haining Xiashi Christ Church",
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
});
