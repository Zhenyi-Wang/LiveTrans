<template>
  <Teleport to="body">
    <Transition name="notification" appear>
      <div
        v-if="show"
        :class="['notification', `notification--${type}`]"
        :style="{...positionClasses, ...customStyle}"
      >
        <div class="notification__content">
          <div class="notification__icon">
            <FontAwesomeIcon :icon="icon" />
          </div>
          <div class="notification__text">
            {{ message }}
          </div>
        </div>
        <button
          v-if="closable"
          class="notification__close"
          @click="close"
        >
          <FontAwesomeIcon :icon="'times'" />
        </button>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import {
  faInfoCircle,
  faCheckCircle,
  faExclamationTriangle,
  faTimesCircle,
  faTimes
} from '@fortawesome/free-solid-svg-icons'
import { library } from '@fortawesome/fontawesome-svg-core'

// 添加图标到库
library.add(
  faInfoCircle,
  faCheckCircle,
  faExclamationTriangle,
  faTimesCircle,
  faTimes
)

const props = defineProps({
  message: {
    type: String,
    required: true
  },
  type: {
    type: String,
    default: 'info',
    validator: (value) => ['info', 'success', 'warning', 'error'].includes(value)
  },
  duration: {
    type: Number,
    default: 2000
  },
  closable: {
    type: Boolean,
    default: true
  },
  position: {
    type: String,
    default: 'top-right',
    validator: (value) => ['top-right', 'top-left', 'bottom-right', 'bottom-left', 'top-center', 'bottom-center'].includes(value)
  },
  customStyle: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['close'])

const show = ref(true)
let timer = null

const icon = computed(() => {
  const icons = {
    info: 'info-circle',
    success: 'check-circle',
    warning: 'exclamation-triangle',
    error: 'times-circle'
  }
  return icons[props.type] || icons.info
})

const positionClasses = computed(() => {
  const positions = {
    'top-right': { top: '20px', right: '20px' },
    'top-left': { top: '20px', left: '20px' },
    'bottom-right': { bottom: '20px', right: '20px' },
    'bottom-left': { bottom: '20px', left: '20px' },
    'top-center': { top: '20px', left: '50%', transform: 'translateX(-50%)' },
    'bottom-center': { bottom: '20px', left: '50%', transform: 'translateX(-50%)' }
  }
  return positions[props.position] || positions['top-right']
})

const close = () => {
  show.value = false
  emit('close')
}

const startTimer = () => {
  if (props.duration > 0) {
    timer = setTimeout(() => {
      close()
    }, props.duration)
  }
}

const clearTimer = () => {
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
}

onMounted(() => {
  startTimer()
})

onUnmounted(() => {
  clearTimer()
})
</script>

<style scoped>
.notification {
  position: fixed;
  z-index: 10000;
  min-width: 300px;
  max-width: 500px;
  padding: 16px 20px;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  background: white;
  border: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 14px;
  line-height: 1.5;
}

.notification__content {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
}

.notification__icon {
  font-size: 18px;
  flex-shrink: 0;
}

.notification__text {
  flex: 1;
  color: #2c3e50;
  white-space: pre-line;
  line-height: 1.4;
}

.notification__close {
  background: none;
  border: none;
  font-size: 16px;
  cursor: pointer;
  color: #718096;
  padding: 4px;
  margin-left: 12px;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.notification__close:hover {
  background: #f7fafc;
  color: #2c3e50;
}

/* 类型样式 */
.notification--info {
  border-left: 4px solid #4299e1;
}

.notification--info .notification__icon {
  color: #4299e1;
}

.notification--success {
  border-left: 4px solid #48bb78;
}

.notification--success .notification__icon {
  color: #48bb78;
}

.notification--warning {
  border-left: 4px solid #ed8936;
}

.notification--warning .notification__icon {
  color: #ed8936;
}

.notification--error {
  border-left: 4px solid #f56565;
}

.notification--error .notification__icon {
  color: #f56565;
}

/* 暗色主题 */
.dark .notification {
  background: #2d3748;
  border-color: #4a5568;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.dark .notification__text {
  color: #e2e8f0;
}

.dark .notification__close {
  color: #a0aec0;
}

.dark .notification__close:hover {
  background: #4a5568;
  color: #e2e8f0;
}

/* 动画 */
.notification-enter-active,
.notification-leave-active {
  transition: all 0.3s ease;
}

.notification-enter-from {
  opacity: 0;
}

.notification-leave-to {
  opacity: 0;
}

/* 顶部中心位置的特殊动画 */
.notification[style*="translateX(-50%)"] {
  transform-origin: top center;
}

.notification[style*="translateX(-50%)"].notification-enter-from {
  transform: translateX(-50%) translateY(-100vh);
}

.notification[style*="translateX(-50%)"].notification-leave-to {
  transform: translateX(-50%) translateY(-20px);
}

/* 响应式 */
@media (max-width: 768px) {
  .notification {
    min-width: 280px;
    max-width: calc(100vw - 40px);
    font-size: 13px;
    padding: 14px 16px;
  }
}
</style>