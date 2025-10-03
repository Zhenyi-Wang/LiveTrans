import { ref, watch, nextTick } from 'vue'
import { useThrottleFn } from '@vueuse/core'

export function useScrollSync(configSyncScroll, configAutoScroll) {
  const isScrolling = ref(false)
  const scrollSource = ref('') // 'chinese' 或 'english'
  const scrollEndSyncTimer = ref(null)

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

  // 滚动结束后的最终同步
  const finalizeScrollSync = (sourceElement, targetSelector) => {
    if (scrollEndSyncTimer.value) {
      clearTimeout(scrollEndSyncTimer.value)
    }

    // 延迟一点时间确保滚动完全结束
    scrollEndSyncTimer.value = setTimeout(() => {
      if (configSyncScroll.value) {
        const targetElement = document.querySelector(targetSelector)
        if (targetElement && sourceElement) {
          const sourceScrollHeight = sourceElement.scrollHeight - sourceElement.clientHeight
          const targetScrollHeight = targetElement.scrollHeight - targetElement.clientHeight

          if (sourceScrollHeight > 0 && targetScrollHeight > 0) {
            const scrollRatio = sourceElement.scrollTop / sourceScrollHeight
            const targetScrollTop = scrollRatio * targetScrollHeight

            // 最终精确同步
            targetElement.scrollTop = targetScrollTop
          }
        }
      }
    }, 100)
  }

  // 联动滚动函数
  const syncScroll = (sourceElement, targetSelector) => {
    if (isScrolling.value) return

    isScrolling.value = true
    const targetElement = document.querySelector(targetSelector)

    console.log('Sync scroll:', {
      targetSelector,
      targetElement: !!targetElement,
      sourceElement: !!sourceElement,
      sourceScrollTop: sourceElement?.scrollTop,
      configSyncScroll: configSyncScroll.value
    })

    if (targetElement && sourceElement) {
      // 计算滚动比例
      const sourceScrollHeight = sourceElement.scrollHeight - sourceElement.clientHeight
      const targetScrollHeight = targetElement.scrollHeight - targetElement.clientHeight

      if (sourceScrollHeight > 0 && targetScrollHeight > 0) {
        const scrollRatio = sourceElement.scrollTop / sourceScrollHeight
        const targetScrollTop = scrollRatio * targetScrollHeight

        console.log('Scroll calculation:', {
          sourceScrollHeight,
          targetScrollHeight,
          scrollRatio,
          targetScrollTop
        })

        // 直接设置滚动位置，不使用平滑滚动（避免冲突）
        targetElement.scrollTop = targetScrollTop

        // 使用 requestAnimationFrame 确保滚动位置准确
        requestAnimationFrame(() => {
          targetElement.scrollTop = targetScrollTop
        })
      }
    }

    // 减少阻塞时间，快速重置状态
    setTimeout(() => {
      isScrolling.value = false
    }, 50)
  }

  // 节流处理的滚动事件
  const onChineseScrollThrottled = (event) => {
    if (!isScrolling.value && configSyncScroll.value) {
      console.log('Chinese scroll triggered, configSyncScroll:', configSyncScroll.value)
      scrollSource.value = 'chinese'
      syncScroll(event.target, '.english-article')
      // 触发最终同步
      finalizeScrollSync(event.target, '.english-article')
    }
  }

  const onEnglishScrollThrottled = (event) => {
    if (!isScrolling.value && configSyncScroll.value) {
      console.log('English scroll triggered, configSyncScroll:', configSyncScroll.value)
      scrollSource.value = 'english'
      syncScroll(event.target, '.chinese-article')
      // 触发最终同步
      finalizeScrollSync(event.target, '.chinese-article')
    }
  }

  // 使用节流函数包装滚动事件（减少节流间隔，提高响应性）
  const onChineseScroll = useThrottleFn(onChineseScrollThrottled, 16)
  const onEnglishScroll = useThrottleFn(onEnglishScrollThrottled, 16)

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
    smoothScrollTo,
    scrollToBottom,
    syncScroll,
    finalizeScrollSync,
    onChineseScroll,
    onEnglishScroll,
    watchDataAndScroll
  }
}