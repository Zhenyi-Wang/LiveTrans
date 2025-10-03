import { ref, computed } from 'vue'
import { useStorage, useToggle, useDark } from "@vueuse/core"

export function useAppConfig() {
  // 主题切换
  const isDark = useDark({
    selector: "body",
    attribute: "class",
    valueDark: "dark",
    valueLight: "",
  })
  const toggleDark = useToggle(isDark)

  // 菜单状态
  const showMenu = ref(false)
  const toggleMenu = useToggle(showMenu)

  // 配置项
  const configAutoScroll = useStorage("config-auto-scroll", true)
  const toggleAutoScroll = useToggle(configAutoScroll)

  const configShowText = useStorage("config-show-text", true)
  const toggleShowText = useToggle(configShowText)

  const configShowTextOpti = useStorage("config-show-text-opti", true)
  const toggleShowTextOpti = useToggle(configShowTextOpti)

  const configShowTextEn = useStorage("config-show-text-en", true)
  const toggleShowTextEn = useToggle(configShowTextEn)

  const configSyncScroll = useStorage("config-sync-scroll", true)
  const toggleSyncScroll = useToggle(configSyncScroll)

  const configParagraphLength = useStorage("config-paragraph-length", 300)
  const configChineseFontSize = useStorage("config-chinese-font-size", 1.3)
  const configEnglishFontSize = useStorage("config-english-font-size", 1.2)

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