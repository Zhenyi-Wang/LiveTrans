<template>
  <div :class="sectionClasses" class="content-section">
    <SectionHeader
      :title="title"
      :is-fullscreen="isFullscreen"
      :font-size="fontSize"
      :language="language"
      :header-classes="headerClasses"
      :auto-scroll="autoScroll"
      @fullscreen="$emit('fullscreen')"
      @font-size-change="$emit('font-size-change', $event)"
      @toggle-auto-scroll="$emit('toggle-auto-scroll')"
      @scroll-to-bottom="$emit('scroll-to-bottom')"
    />

    <div class="content-area" :class="`${language}-content`">
      <div v-if="isWaitingForService" class="welcome-message">
        {{ welcomeMessage }}
        <div class="report-hint" @click="$emit('report')">
          <FontAwesomeIcon icon="comment-dots" />
          <span>{{ language === 'chinese' ? '如有问题请点击汇报' : 'Spot a problem? Report it' }}</span>
        </div>
      </div>

      <ParagraphDisplay
        :language="language"
        :paragraphs="paragraphs"
        :font-size="fontSize"
        @scroll="$emit('scroll', $event)"
      />

      <CurrentInput
        v-if="!isWaitingForService"
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
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { faCommentDots } from '@fortawesome/free-solid-svg-icons'
import { library } from '@fortawesome/fontawesome-svg-core'
import SectionHeader from './SectionHeader.vue'
import ParagraphDisplay from './ParagraphDisplay.vue'
import CurrentInput from './CurrentInput.vue'

library.add(faCommentDots)

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
  },
  autoScroll: {
    type: Boolean,
    required: true
  }
})

defineEmits(['fullscreen', 'font-size-change', 'scroll', 'toggle-auto-scroll', 'scroll-to-bottom', 'report'])

const welcomeMessage = computed(() => {
  return props.language === 'chinese'
    ? '欢迎来到海宁市硖石基督教堂！😊 我们很高兴您的到来。请耐心等待，我们即将开始...'
    : 'Welcome to Haining Xiashi Christian Church!😊 We\'re delighted to have you here. Please be patient and wait while we begin...'
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

.report-hint {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 12px;
  font-size: 0.9em;
  color: var(--primary-color, #00adb5);
  cursor: pointer;
  opacity: 0.85;
  transition: all 0.2s ease;
}

.report-hint:hover {
  opacity: 1;
  text-decoration: underline;
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