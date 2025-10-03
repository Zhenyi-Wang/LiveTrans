<template>
  <div :class="articleClasses" class="article-display" @scroll="$emit('scroll', $event)">
    <div
      v-for="(paragraph, index) in processedParagraphs"
      :key="`${language}-${index}`"
      class="article-paragraph"
      :class="{ 'translating': language === 'english' && !paragraph }"
    >
      <div class="paragraph-content" :class="contentClasses" :style="{ fontSize: fontSize + 'rem' }">
        <!-- 中文段落显示 -->
        <template v-if="language === 'chinese'">
          <TransitionGroup name="segment" tag="div" class="segments-container">
            <span
              v-for="(segment, segIndex) in paragraph"
              :key="`cn-${index}-${segIndex}`"
              :class="[
                segment.isOptimized ? 'optimized-text' : 'unoptimized-text',
                'text-segment'
              ]"
            >
              {{ segment.text }}
            </span>
          </TransitionGroup>
        </template>

        <!-- 英文段落显示 -->
        <template v-else>
          <template v-if="paragraph">
            <span
              v-for="(segment, segIndex) in paragraph"
              :key="segment.id || `en-${index}-${segIndex}`"
              class="text-segment english-segment"
              :class="{
                'translating': segment.isTranslating,
                'has-translation': segment.hasContent
              }"
            >
              <span v-if="segment.hasContent" class="translation-content">{{ segment.text }} </span>
              <span v-if="segment.isTranslating" class="translating-dots">... </span>
              <span v-if="segment.hasContent && segIndex < paragraph.length - 1"> </span>
            </span>
          </template>
        </template>
      </div>
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
  paragraphs: {
    type: Array,
    required: true
  },
  fontSize: {
    type: Number,
    required: true
  }
})

defineEmits(['scroll'])

const articleClasses = computed(() => ({
  'chinese-article': props.language === 'chinese',
  'english-article': props.language === 'english',
  'flex-article': true
}))

const contentClasses = computed(() => ({
  'english-content': props.language === 'english'
}))

const processedParagraphs = computed(() => {
  return props.paragraphs.map(paragraph => {
    if (!paragraph) return null

    if (props.language === 'chinese') {
      return paragraph.filter(segment => segment && segment.text)
    } else {
      return paragraph.map(segment => ({
        ...segment,
        hasContent: segment && (segment.text || '').trim().length > 0,
        isTranslating: segment && !segment.hasTranslation && !segment.text
      }))
    }
  })
})
</script>

<style scoped>
/* 确保所有元素使用border-box */
*, *::before, *::after {
  box-sizing: border-box;
}

.article-display {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  margin-bottom: 1em;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  margin: 0;
  padding: 0 0 1em 0;
  padding-left: calc(50% - 300px);
  padding-right: calc(50% - 300px);
  box-sizing: border-box;
  /* 优化手机触屏滚动 */
  -webkit-overflow-scrolling: touch;
  /* 禁用CSS平滑滚动，避免与JS冲突 */
  scroll-behavior: auto;
}

.article-paragraph {
  margin-bottom: 1.5em;
  position: relative;
  transition: all 0.3s ease;
}

.article-paragraph.translating {
  opacity: 0.9;
}

.paragraph-content {
  color: #2c3e50;
  font-size: 1.3rem;
  line-height: 1.8;
  text-align: justify;
  text-indent: 2em;
  font-weight: 400;
  margin-bottom: 0.5em;
  word-wrap: break-word;
  white-space: pre-wrap;
}

.english-content .paragraph-content {
  color: var(--text-color);
  text-align: left;
  font-size: var(--english-font-size, 1.2rem);
  text-indent: 0;
}

.text-segment {
  display: inline;
}

.optimized-text {
  color: #2c3e50;
  font-weight: 400;
}

.unoptimized-text {
  color: #666;
  font-style: italic;
  opacity: 0.8;
}

.english-segment {
  line-height: 1.6;
  min-height: 1.6em;
}

.english-segment.pending {
  color: #ccc;
  font-style: italic;
}

.translating-dots {
  animation: dots 1.5s infinite;
}

.translation-content {
  display: inline;
  animation: fadeInUp 1.5s ease-out forwards;
}

@keyframes dots {
  0%, 20% { opacity: 0; }
  50% { opacity: 1; }
  80%, 100% { opacity: 0; }
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

.segments-container {
  display: inline;
}

/* 深色主题 */
.dark .paragraph-content {
  color: #e2e8f0;
}

.dark .optimized-text {
  color: #e2e8f0;
}

.dark .unoptimized-text {
  color: #999;
}


/* 响应式调整 */
@media (max-width: 768px) {
  .article-display {
    padding-left: 2em;
    padding-right: 2em;
    box-sizing: border-box;
  }

  .paragraph-content {
    font-size: var(--chinese-font-size, 1.1rem);
    text-indent: 1.5em;
  }

  .english-content .paragraph-content {
    font-size: var(--english-font-size, 1rem);
  }

  .article-paragraph {
    margin-bottom: 1.2em;
  }
}
</style>