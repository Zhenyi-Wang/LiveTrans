import { ref, computed } from 'vue'
import { useStorage, useToggle, useDark } from "@vueuse/core"
import { useGlobalNotification } from './useNotification'

export function useAppConfig() {
  const { success } = useGlobalNotification()

  // 获取当前语言设置（这里可以进一步优化，从全局状态获取）
  const getCurrentLanguage = () => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('language') || 'chinese'
    }
    return 'chinese'
  }

  // 获取多语言文本
  const t = (chineseText: string, englishText: string) => {
    return getCurrentLanguage() === 'english' ? englishText : chineseText
  }

  // 主题切换
  const isDark = useDark({
    selector: "body",
    attribute: "class",
    valueDark: "dark",
    valueLight: "",
    initialValue: "dark",
  })
  const toggleDark = useToggle(isDark)

  // 菜单状态
  const showMenu = ref(false)
  const toggleMenu = useToggle(showMenu)

  // 配置项
  const configAutoScroll = useStorage("config-auto-scroll", true)
  const toggleAutoScroll = () => {
    configAutoScroll.value = !configAutoScroll.value
    // 显示状态提示（中英文分行显示）
    const chineseMessage = configAutoScroll.value ? '自动滚动已开启' : '自动滚动已关闭'
    const englishMessage = configAutoScroll.value ? 'Auto-scroll enabled' : 'Auto-scroll disabled'
    const message = `${chineseMessage}\n${englishMessage}`
    success(message, { position: 'top-center' })
  }

  const configShowText = useStorage("config-show-text", true)
  const toggleShowText = useToggle(configShowText)

  const configShowTextOpti = useStorage("config-show-text-opti", true)
  const toggleShowTextOpti = useToggle(configShowTextOpti)

  const configShowTextEn = useStorage("config-show-text-en", true)
  const toggleShowTextEn = useToggle(configShowTextEn)

  const configSyncScroll = useStorage("config-sync-scroll", true)
  const toggleSyncScroll = useToggle(configSyncScroll)

  const configParagraphLength = useStorage("config-paragraph-length", 150)
  const configChineseFontSize = useStorage("config-chinese-font-size", 1.1)
  const configEnglishFontSize = useStorage("config-english-font-size", 1.1)

  // 计算属性
  const maxParagraphLength = computed(() => configParagraphLength.value)

  // CSS 变量
  const cssVariables = computed(() => ({
    '--chinese-font-size': configChineseFontSize.value + 'rem',
    '--english-font-size': configEnglishFontSize.value + 'rem',
    '--chinese-input-height': (configChineseFontSize.value * 1.8) + 'em',
    '--english-input-height': (configEnglishFontSize.value * 1.8) + 'em'
  }))

  return {
    // 主题
    isDark,
    toggleDark,

    // 菜单
    showMenu,
    toggleMenu,

    // 配置项
    configAutoScroll,
    toggleAutoScroll,
    configShowText,
    toggleShowText,
    configShowTextOpti,
    toggleShowTextOpti,
    configShowTextEn,
    toggleShowTextEn,
    configSyncScroll,
    toggleSyncScroll,
    configParagraphLength,
    configChineseFontSize,
    configEnglishFontSize,

    // 计算属性
    maxParagraphLength,
    cssVariables
  }
}