<template>
  <div :class="sectionClasses" class="content-section">
    <SectionHeader
      :title="title"
      :is-fullscreen="isFullscreen"
      :font-size="fontSize"
      :language="language"
      :header-classes="headerClasses"
      @fullscreen="$emit('fullscreen')"
      @font-size-change="$emit('font-size-change', $event)"
    />

    <div class="content-area" :class="`${language}-content`">
      <div v-if="isWaitingForService" class="welcome-message">
        {{ welcomeMessage }}
      </div>

      <ParagraphDisplay
        :language="language"
        :paragraphs="paragraphs"
        :font-size="fontSize"
        @scroll="$emit('scroll', $event)"
      />

      <CurrentInput
        :language="language"
        :current-segment="currentSegment"
        :is-english="language === 'english'"
        :font-size="fontSize"
        :last-current-en="lastCurrentEn"
      />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import SectionHeader from './SectionHeader.vue'
import ParagraphDisplay from './ParagraphDisplay.vue'
import CurrentInput from './CurrentInput.vue'

const props = defineProps({
  language: {
    type: String,
    required: true,
    validator: (value) => ['chinese', 'english'].includes(value)
  },
  title: {
    type: String,
    required: true
  },
  paragraphs: {
    type: Array,
    required: true
  },
  currentSegment: {
    type: Object,
    required: true
  },
  fontSize: {
    type: Number,
    required: true
  },
  sectionClasses: {
    type: Object,
    required: true
  },
  headerClasses: {
    type: Object,
    required: true
  },
  isFullscreen: {
    type: Boolean,
    required: true
  },
  isWaitingForService: {
    type: Boolean,
    required: true
  },
  lastCurrentEn: {
    type: String,
    default: ''
  }
})

defineEmits(['fullscreen', 'font-size-change', 'scroll'])

const welcomeMessage = computed(() => {
  return props.language === 'chinese'
    ? '欢迎来到海宁市硖石基督教堂！😊 我们很高兴您的到来。请耐心等待，我们即将开始...'
    : 'Welcome to Haining Xiashi Christ Church!😊 We\'re delighted to have you here. Please be patient and wait while we begin...'
})
</script>

<style scoped>
.content-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-bottom: 2px solid #e0e0e0;
  transition: all 0.3s ease;
}

.content-section.fullscreen {
  flex: 1;
  height: 100%;
  border-bottom: none;
}

.content-section.hidden {
  display: none;
}

.content-area {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.chinese-content {
  --chinese-font-size: v-bind(fontSize + 'rem');
}

.english-content {
  --english-font-size: v-bind(fontSize + 'rem');
}

.welcome-message {
  text-align: center;
  padding: 2em 20px;
  color: #666;
  font-size: var(--chinese-font-size, 1.3rem);
  line-height: 1.6;
}

.english-content .welcome-message {
  font-size: var(--english-font-size, 1.2rem);
}

.dark .welcome-message {
  color: #999;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .welcome-message {
    padding: 1.5em 15px;
    font-size: var(--chinese-font-size, 1.3rem);
  }

  .english-content .welcome-message {
    font-size: var(--english-font-size, 1.2rem);
  }
}
</style>