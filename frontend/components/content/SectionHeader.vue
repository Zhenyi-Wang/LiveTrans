<template>
  <div class="section-header" :class="headerClasses">
    <div class="header-left">
      <div class="fullscreen-btn">
        <button class="font-size-btn" @click="$emit('fullscreen')">
          <FontAwesomeIcon :icon="fullscreenIcon" />
        </button>
      </div>
      <h3 class="section-title clickable" @click="$emit('fullscreen')">{{ title }}</h3>
    </div>

    <div class="header-right">
      <div class="auto-scroll-controls">
        <button
          class="font-size-btn auto-scroll-btn"
          :class="{ 'active': autoScroll }"
          @click="handleAutoScrollClick"
          title="自动滚动 | Auto Scroll"
        >
          <FontAwesomeIcon icon="arrow-down" />
        </button>
      </div>
      <div class="font-size-controls">
        <button
          class="font-size-btn"
          @click="$emit('font-size-change', Math.max(0.8, fontSize - 0.1))"
          :disabled="fontSize <= 0.8"
        >
          <FontAwesomeIcon icon="minus" />
        </button>
        <button
          class="font-size-btn"
          @click="$emit('font-size-change', Math.min(2.0, fontSize + 0.1))"
          :disabled="fontSize >= 2.0"
        >
          <FontAwesomeIcon icon="plus" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick } from 'vue'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import {
  faExpand,
  faCompress,
  faMinus,
  faPlus,
  faArrowDown
} from '@fortawesome/free-solid-svg-icons'
import { library } from '@fortawesome/fontawesome-svg-core'

// 添加图标到库
library.add(
  faExpand,
  faCompress,
  faMinus,
  faPlus,
  faArrowDown
)

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  isFullscreen: {
    type: Boolean,
    required: true
  },
  fontSize: {
    type: Number,
    required: true
  },
  language: {
    type: String,
    required: true
  },
  headerClasses: {
    type: Object,
    required: true
  },
  autoScroll: {
    type: Boolean,
    required: true
  }
})

const emit = defineEmits(['fullscreen', 'font-size-change', 'toggle-auto-scroll', 'scroll-to-bottom'])

const fullscreenIcon = computed(() => props.isFullscreen ? 'compress' : 'expand')

const handleAutoScrollClick = () => {
  emit('toggle-auto-scroll')
  // 如果是激活自动滚动，立即触发一次滚动
  nextTick(() => {
    emit('scroll-to-bottom')
  })
}
</script>

<style scoped>
.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-bottom: 1px solid #dee2e6;
  flex-shrink: 0;
  transition: all 0.3s ease;
}

.section-header.disabled {
  opacity: 0.5;
  pointer-events: none;
}

.section-header.fullscreen {
  background: linear-gradient(135deg, #00adb5 0%, #00c4cc 100%);
  color: white;
}

.header-left,
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.section-title {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: inherit;
  transition: all 0.2s ease;
}

.section-title.clickable {
  cursor: pointer;
  user-select: none;
}

.section-title.clickable:hover {
  transform: scale(1.05);
}

.fullscreen-btn {
  display: flex;
  align-items: center;
}

.auto-scroll-controls {
  display: flex;
  align-items: center;
  margin-right: 8px;
}

.font-size-controls {
  display: flex;
  align-items: center;
  gap: 8px;
}

.font-size-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 50%;
  background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
  color: var(--primary-color, #00adb5);
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  font-size: 0.9rem;
}

.font-size-btn:hover:not(:disabled) {
  transform: scale(1.1);
  box-shadow: 0 4px 8px rgba(0, 173, 181, 0.3);
  background: linear-gradient(135deg, var(--primary-color, #00adb5) 0%, var(--primary-hover, #00c4cc) 100%);
  color: white;
}

.font-size-btn:active:not(:disabled) {
  transform: scale(0.95);
}

.font-size-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.section-header.fullscreen .font-size-btn {
  background: rgba(255, 255, 255, 0.2);
  color: white;
}

.section-header.fullscreen .font-size-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.3);
}

/* 自动滚动按钮样式 */
.auto-scroll-btn {
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  color: #6c757d;
  border: 2px solid #dee2e6;
  position: relative;
  overflow: hidden;
}

.auto-scroll-btn:hover {
  background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
  border-color: var(--primary-color, #00adb5);
  color: var(--primary-color, #00adb5);
  transform: scale(1.05);
}

.auto-scroll-btn.active {
  background: linear-gradient(135deg, var(--primary-color, #00adb5) 0%, var(--primary-hover, #00c4cc) 100%);
  color: white;
  border-color: var(--primary-color, #00adb5);
  box-shadow: 0 4px 12px rgba(0, 173, 181, 0.4);
  animation: pulse 2s infinite;
}

.auto-scroll-btn.active:hover {
  background: linear-gradient(135deg, #0096a3 0%, #00adb5 100%);
  transform: scale(1.05);
  box-shadow: 0 6px 16px rgba(0, 173, 181, 0.5);
}

@keyframes pulse {
  0%, 100% {
    box-shadow: 0 4px 12px rgba(0, 173, 181, 0.4);
  }
  50% {
    box-shadow: 0 4px 12px rgba(0, 173, 181, 0.6);
  }
}

.section-header.fullscreen .auto-scroll-btn {
  background: rgba(255, 255, 255, 0.15);
  color: rgba(255, 255, 255, 0.7);
  border-color: rgba(255, 255, 255, 0.3);
}

.section-header.fullscreen .auto-scroll-btn:hover {
  background: rgba(255, 255, 255, 0.25);
  border-color: rgba(255, 255, 255, 0.6);
  color: rgba(255, 255, 255, 0.9);
}

.section-header.fullscreen .auto-scroll-btn.active {
  background: rgba(255, 255, 255, 0.9);
  color: var(--primary-color, #00adb5);
  border-color: rgba(255, 255, 255, 0.9);
  box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
}

.section-header.fullscreen .auto-scroll-btn.active:hover {
  background: rgba(255, 255, 255, 1);
  color: #0096a3;
  box-shadow: 0 6px 16px rgba(255, 255, 255, 0.4);
}

/* 深色主题 */
.dark .section-header {
  background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
  border-bottom-color: #4a5568;
}

.dark .section-title {
  color: #e2e8f0;
}

.dark .font-size-btn {
  background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
  color: #00adb5;
}

.dark .font-size-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, #00adb5 0%, #00c4cc 100%);
  color: white;
}

/* 深色主题下自动滚动按钮样式 */
.dark .auto-scroll-btn {
  background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
  color: #a0aec0;
  border-color: #4a5568;
}

.dark .auto-scroll-btn:hover {
  background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
  border-color: var(--primary-color, #00adb5);
  color: var(--primary-color, #00adb5);
}

.dark .auto-scroll-btn.active {
  background: linear-gradient(135deg, var(--primary-color, #00adb5) 0%, var(--primary-hover, #00c4cc) 100%);
  color: white;
  border-color: var(--primary-color, #00adb5);
  box-shadow: 0 4px 12px rgba(0, 173, 181, 0.4);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .section-header {
    padding: 10px 15px;
  }

  .section-title {
    font-size: 1rem;
  }

  .font-size-btn {
    width: 28px;
    height: 28px;
    font-size: 0.8rem;
  }
}
</style>