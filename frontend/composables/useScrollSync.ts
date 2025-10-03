import { ref, watch, nextTick, computed } from 'vue'
import { useThrottleFn } from '@vueuse/core'

// 安全的触屏设备检测（支持SSR）
const isTouchDevice = () => {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return false // SSR环境默认返回false
  }
  return 'ontouchstart' in window || navigator.maxTouchPoints > 0 || navigator.msMaxTouchPoints > 0
}

export function useScrollSync(configSyncScroll, configAutoScroll) {
  const isScrolling = ref(false)
  const scrollSource = ref('') // 'chinese' 或 'english'
  const scrollEndSyncTimer = ref(null)
  const finalSyncTimer = ref(null) // 最终同步定时器
  const lastScrollTime = ref(0) // 最后一次滚动时间

  // 触屏设备检测缓存（客户端安全）
  const touchDevice = ref(false)

  // 在客户端初始化时检测触屏设备
  if (process.client) {
    touchDevice.value = isTouchDevice()
  }

  // 平滑滚动函数 - 0.5秒滚动动画
  const smoothScrollTo = (element, targetScrollTop, duration = 500) => {
    const startScrollTop = element.scrollTop
    const distance = targetScrollTop - startScrollTop
    const startTime = performance.now()

    const animateScroll = (currentTime) => {
      const elapsed = currentTime - startTime
      const progress = Math.min(elapsed / duration, 1)

      // 使用 ease-in-out 缓动函数
      const easeProgress = progress < 0.5
        ? 2 * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 2) / 2

      element.scrollTop = startScrollTop + (distance * easeProgress)

      if (progress < 1) {
        requestAnimationFrame(animateScroll)
      }
    }

    requestAnimationFrame(animateScroll)
  }

  // 自动滚动 - 分别控制中文和英文区域
  const scrollToBottom = () => {
    // 滚动中文区域到底部
    const chineseContent = document.querySelector('.chinese-article.article-display')
    if (chineseContent) {
      const targetScrollTop = chineseContent.scrollHeight + 1000
      smoothScrollTo(chineseContent, targetScrollTop, 500)
    }

    // 滚动英文区域到底部
    const englishContent = document.querySelector('.english-article.article-display')
    if (englishContent) {
      const targetScrollTop = englishContent.scrollHeight + 1000
      smoothScrollTo(englishContent, targetScrollTop, 500)
    }
  }

  // 精确的最终位置同步
  const finalPositionSync = (sourceElement, targetSelector) => {
    if (!configSyncScroll.value || !sourceElement) return

    const targetElement = document.querySelector(targetSelector)
    if (!targetElement) return

    const sourceScrollHeight = sourceElement.scrollHeight - sourceElement.clientHeight
    const targetScrollHeight = targetElement.scrollHeight - targetElement.clientHeight

    if (sourceScrollHeight > 0 && targetScrollHeight > 0) {
      const scrollRatio = sourceElement.scrollTop / sourceScrollHeight
      const targetScrollTop = scrollRatio * targetScrollHeight

      // 使用 requestAnimationFrame 确保精确同步
      requestAnimationFrame(() => {
        targetElement.scrollTop = targetScrollTop
      })
    }
  }

  // 优化的联动滚动函数
  const syncScroll = (sourceElement, targetSelector) => {
    if (isScrolling.value) return

    isScrolling.value = true

    const targetElement = document.querySelector(targetSelector)

    if (targetElement && sourceElement) {
      // 计算滚动比例
      const sourceScrollHeight = sourceElement.scrollHeight - sourceElement.clientHeight
      const targetScrollHeight = targetElement.scrollHeight - targetElement.clientHeight

      if (sourceScrollHeight > 0 && targetScrollHeight > 0) {
        const scrollRatio = sourceElement.scrollTop / sourceScrollHeight
        const targetScrollTop = scrollRatio * targetScrollHeight

        // 触屏设备使用更简单的同步策略
        if (touchDevice.value) {
          // 触屏设备：直接设置滚动位置，避免动画
          targetElement.scrollTop = targetScrollTop
        } else {
          // 桌面设备：使用requestAnimationFrame确保平滑
          requestAnimationFrame(() => {
            targetElement.scrollTop = targetScrollTop
          })
        }
      }
    }

    // 根据设备类型调整状态重置时间
    const resetDelay = touchDevice.value ? 100 : 50
    setTimeout(() => {
      isScrolling.value = false
    }, resetDelay)
  }

  // 滚动结束检测和最终同步
  const scheduleFinalSync = (sourceElement, targetSelector) => {
    // 更新最后滚动时间
    lastScrollTime.value = Date.now()

    // 清除之前的定时器
    if (scrollEndSyncTimer.value) {
      clearTimeout(scrollEndSyncTimer.value)
    }

    // 设置新的滚动结束检测
    scrollEndSyncTimer.value = setTimeout(() => {
      // 检查是否真的停止了滚动（距离最后一次滚动超过150ms）
      const now = Date.now()
      if (now - lastScrollTime.value >= 150) {
        // 滚动确实停止了，执行最终精确同步
        finalPositionSync(sourceElement, targetSelector)
      }
    }, 150)
  }

  // 优化的节流处理滚动事件
  const onChineseScrollThrottled = (event) => {
    if (!isScrolling.value && configSyncScroll.value) {
      scrollSource.value = 'chinese'
      syncScroll(event.target, '.english-article')

      // 安排最终同步
      scheduleFinalSync(event.target, '.english-article')
    }
  }

  const onEnglishScrollThrottled = (event) => {
    if (!isScrolling.value && configSyncScroll.value) {
      scrollSource.value = 'english'
      syncScroll(event.target, '.chinese-article')

      // 安排最终同步
      scheduleFinalSync(event.target, '.chinese-article')
    }
  }

  // 创建节流函数（统一使用50ms间隔，对手机和桌面都友好）
  const onChineseScroll = useThrottleFn(onChineseScrollThrottled, 50)
  const onEnglishScroll = useThrottleFn(onEnglishScrollThrottled, 50)

  // 监听数据变化并自动滚动
  const watchDataAndScroll = (currentSegment, confirmedSegments) => {
    watch(
      [currentSegment, confirmedSegments],
      () => {
        if (configAutoScroll.value) {
          scrollToBottom()
        }
      },
      { deep: true }
    )
  }

  return {
    isScrolling,
    scrollSource,
    touchDevice,
    smoothScrollTo,
    scrollToBottom,
    syncScroll,
    finalPositionSync,
    onChineseScroll,
    onEnglishScroll,
    watchDataAndScroll
  }
}