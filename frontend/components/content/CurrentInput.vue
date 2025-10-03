<template>
  <div class="current-input" :class="inputClasses">
    <div class="paragraph-content current-input-content" :class="contentClasses" :style="{ fontSize: fontSize + 'rem' }">
      <span class="blinking-cursor"> |</span>
      <span class="truncated-text">{{ displayText }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  language: {
    type: String,
    required: true,
    validator: (value) => ['chinese', 'english'].includes(value)
  },
  currentSegment: {
    type: Object,
    required: true
  },
  isEnglish: {
    type: Boolean,
    required: true
  },
  fontSize: {
    type: Number,
    required: true
  },
  lastCurrentEn: {
    type: String,
    default: ''
  }
})

const inputClasses = computed(() => ({
  'english-input': props.isEnglish
}))

const contentClasses = computed(() => ({
  'english-content': props.isEnglish
}))

const displayText = computed(() => {
  if (props.isEnglish) {
    return props.lastCurrentEn || ''
  }
  return props.currentSegment.text || ''
})
</script>

<style scoped>
.current-input {
  padding: 0.5em 20px;
  border-top: 1px solid #e0e0e0;
  background: rgba(248, 249, 250, 0.5);
  backdrop-filter: blur(5px);
  flex-shrink: 0;
  min-height: 2em;
  display: flex;
  align-items: center;
}

.current-input.english-input {
  background: rgba(240, 240, 240, 0.3);
}

.current-input-content {
  color: #666;
  font-style: italic;
  text-indent: 0;
  line-height: 1.8;
  transition: all 0.3s ease;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
  direction: rtl;
  text-align: left;
}

.english-content.current-input-content {
  color: #999;
}

.blinking-cursor {
  color: var(--primary-color, #00adb5);
  font-weight: bold;
  animation: blink 1s infinite;
  margin-right: 2px;
}

.truncated-text {
  color: #666;
}

.truncated-text::after {
  content: "\200E";
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* 深色主题 */
.dark .current-input {
  border-top-color: #4a5568;
  background: rgba(45, 55, 72, 0.5);
}

.dark .current-input-content {
  color: #999;
}

.dark .truncated-text {
  color: #999;
}

.dark .blinking-cursor {
  color: var(--primary-color, #00adb5);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .current-input {
    padding: 0.4em 15px;
    min-height: 1.8em;
  }
}
</style>