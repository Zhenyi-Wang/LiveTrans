<template>
  <div class="segment-display">
    <template v-if="type === 'chinese'">
      <TransitionGroup name="segment" tag="div" class="segments-container">
        <span
          v-for="(segment, index) in segments"
          :key="segment.key || `cn-${index}`"
          :class="segmentClasses(segment)"
        >
          {{ segment.text }}
        </span>
      </TransitionGroup>
    </template>
    <template v-else-if="type === 'english'">
      <div class="segments-container">
        <span
          v-for="(segment, index) in segments"
          :key="segment.id || `en-${index}`"
          class="text-segment english-segment"
          :class="{
            'translating': segment.isTranslating,
            'has-translation': segment.hasContent
          }"
        >
          <span v-if="segment.hasContent" class="translation-content">{{ segment.text }} </span>
          <span v-if="segment.isTranslating" class="translating-dots">... </span>
          <span v-if="segment.hasContent && index < segments.length - 1"> </span>
        </span>
      </div>
    </template>
  </div>
</template>

<script setup>
const props = defineProps({
  segments: {
    type: Array,
    required: true
  },
  type: {
    type: String,
    default: 'chinese',
    validator: (value) => ['chinese', 'english'].includes(value)
  }
})

const segmentClasses = (segment) => {
  if (props.type === 'chinese') {
    return [
      segment.isOptimized ? 'optimized-text' : 'unoptimized-text',
      'text-segment'
    ]
  } else {
    return [
      'text-segment',
      'english-segment',
      {
        'translating': segment.isTranslating,
        'has-translation': segment.hasContent
      }
    ]
  }
}
</script>

<style scoped>
.segments-container {
  display: inline;
}

.text-segment {
  display: inline;
  padding: 2px 4px;
  margin: -2px -4px;
  border-radius: 4px;
}

/* 中文样式 */
.optimized-text {
  color: #0088cc;
  font-weight: 500;
}

.unoptimized-text {
  color: #888;
}

/* 英文样式 */
.english-segment {
  line-height: 1.6;
  min-height: 1.6em;
}

.english-segment.translating {
  color: #ccc;
  font-style: italic;
}

.english-segment.has-translation {
  color: var(--text-color, #2c3e50);
}

/* 翻译内容特殊动画 */
.translation-content {
  display: inline;
  animation: fadeInUp 1.5s ease-out forwards;
}

.translating-dots {
  animation: dots 1.5s infinite;
}

@keyframes dots {
  0%, 20% { opacity: 0; }
  50% { opacity: 1; }
  80%, 100% { opacity: 0; }
}

/* 进入动画 */
.segment-enter-active {
  transition: all 1.5s ease-out;
}

.segment-enter-from {
  opacity: 0;
  transform: translateY(5px);
}

.segment-enter-to {
  opacity: 1;
  transform: translateY(0);
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(5px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 暗色主题 */
.dark .optimized-text {
  color: #66b3ff;
  font-weight: 500;
}

.dark .unoptimized-text {
  color: #999;
}

.dark .english-segment {
  color: var(--text-color, #e2e8f0);
}

.dark .english-segment.translating {
  color: #666;
}
</style>