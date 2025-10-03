import { ref, computed } from 'vue'
import { useStorage } from '@vueuse/core'

export function useFullscreen() {
  const isChineseFullscreen = useStorage('is-chinese-fullscreen', false)
  const isEnglishFullscreen = useStorage('is-english-fullscreen', false)

  const toggleChineseFullscreen = () => {
    if (isChineseFullscreen.value) {
      // 取消全屏
      isChineseFullscreen.value = false
    } else {
      // 设置中文全屏，同时取消英文全屏
      isChineseFullscreen.value = true
      isEnglishFullscreen.value = false
    }
  }

  const toggleEnglishFullscreen = () => {
    if (isEnglishFullscreen.value) {
      // 取消全屏
      isEnglishFullscreen.value = false
    } else {
      // 设置英文全屏，同时取消中文全屏
      isEnglishFullscreen.value = true
      isChineseFullscreen.value = false
    }
  }

  // 获取中文区域的样式类
  const getChineseSectionClasses = computed(() => ({
    'hidden': isEnglishFullscreen.value,
    'fullscreen': isChineseFullscreen.value
  }))

  // 获取英文区域的样式类
  const getEnglishSectionClasses = computed(() => ({
    'hidden': isChineseFullscreen.value,
    'fullscreen': isEnglishFullscreen.value
  }))

  // 获取中文头部样式类
  const getChineseHeaderClasses = computed(() => ({
    'disabled': isEnglishFullscreen.value,
    'fullscreen': isChineseFullscreen.value
  }))

  // 获取英文头部样式类
  const getEnglishHeaderClasses = computed(() => ({
    'disabled': isChineseFullscreen.value,
    'fullscreen': isEnglishFullscreen.value
  }))

  // 获取分隔线样式类
  const getDividerClasses = computed(() => ({
    'hidden': isChineseFullscreen.value || isEnglishFullscreen.value
  }))

  // 获取全屏按钮图标
  const getChineseFullscreenIcon = computed(() =>
    isChineseFullscreen.value ? 'compress' : 'expand'
  )

  const getEnglishFullscreenIcon = computed(() =>
    isEnglishFullscreen.value ? 'compress' : 'expand'
  )

  return {
    // 状态
    isChineseFullscreen,
    isEnglishFullscreen,

    // 方法
    toggleChineseFullscreen,
    toggleEnglishFullscreen,

    // 计算属性
    getChineseSectionClasses,
    getEnglishSectionClasses,
    getChineseHeaderClasses,
    getEnglishHeaderClasses,
    getDividerClasses,
    getChineseFullscreenIcon,
    getEnglishFullscreenIcon
  }
}