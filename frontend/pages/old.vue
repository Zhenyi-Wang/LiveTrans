<script setup>
// Font Awesome 图标在子组件中使用

// 引入composables
const {
  isDark,
  toggleDark,
  showMenu,
  toggleMenu,
  configAutoScroll,
  toggleAutoScroll,
  configShowText,
  configShowTextEn,
  configSyncScroll,
  toggleSyncScroll,
  configParagraphLength,
  configChineseFontSize,
  configEnglishFontSize,
  cssVariables
} = useAppConfig()

const {
  wsConnected,
  currentSegment,
  lastCurrentEn,
  confirmedSegments,
  isWaitingForService
} = useWebSocket(() => {
  // WebSocket 初始化完成后的回调
  if (configAutoScroll.value) {
    scrollToBottom()
  }
})

const { getChineseParagraphs, getEnglishParagraphs } = useParagraphLogic(
  confirmedSegments,
  computed(() => configParagraphLength.value)
)

const {
  onChineseScroll,
  onEnglishScroll,
  scrollToBottom,
  watchDataAndScroll
} = useScrollSync(configSyncScroll, configAutoScroll)

const {
  isChineseFullscreen,
  isEnglishFullscreen,
  getChineseSectionClasses,
  getEnglishSectionClasses,
  getChineseHeaderClasses,
  getEnglishHeaderClasses,
  getDividerClasses,
  toggleChineseFullscreen,
  toggleEnglishFullscreen
} = useFullscreen()

// 监听数据变化并自动滚动
watchDataAndScroll(currentSegment, confirmedSegments)

// 报错反馈弹窗(三入口共用: header按钮/菜单末项/欢迎词)
const showReport = ref(false)
const openReport = () => {
  showMenu.value = false
  showReport.value = true
}
</script>

<template>
  <ClientOnly>
    <div class="optimized-layout" :style="cssVariables">
      <!-- 使用AppHeader组件 -->
      <LayoutAppHeader
        :ws-connected="wsConnected"
        :show-menu="showMenu"
        :is-dark="isDark"
        :config-auto-scroll="configAutoScroll"
        :config-sync-scroll="configSyncScroll"
        :config-paragraph-length="configParagraphLength"
        :config-chinese-font-size="configChineseFontSize"
        :config-english-font-size="configEnglishFontSize"
        :config-show-text="configShowText"
        :config-show-text-en="configShowTextEn"
        @toggle-menu="toggleMenu"
        @report="openReport"
        @toggle-dark="toggleDark"
        @toggle-auto-scroll="toggleAutoScroll"
        @toggle-sync-scroll="toggleSyncScroll"
        @scroll-to-bottom="scrollToBottom"
        @update:config-paragraph-length="configParagraphLength = $event"
        @update:config-chinese-font-size="configChineseFontSize = $event"
        @update:config-english-font-size="configEnglishFontSize = $event"
        @update:config-show-text="configShowText = $event"
        @update:config-show-text-en="configShowTextEn = $event"
      />

      <div class="content-container">
        <!-- 中文内容区域 -->
        <ContentSection
        language="chinese"
        title="中文 | Chinese"
        :paragraphs="getChineseParagraphs"
        :current-segment="currentSegment"
        :font-size="configChineseFontSize"
          :section-classes="getChineseSectionClasses"
        :header-classes="getChineseHeaderClasses"
        :is-fullscreen="isChineseFullscreen"
        :is-waiting-for-service="isWaitingForService"
        :last-current-en="lastCurrentEn"
        :auto-scroll="configAutoScroll"
        @report="openReport"
        @fullscreen="toggleChineseFullscreen"
        @font-size-change="configChineseFontSize = $event"
        @scroll="onChineseScroll"
        @toggle-auto-scroll="toggleAutoScroll"
        @scroll-to-bottom="scrollToBottom"
      />

      <!-- 分隔线 -->
      <ContentSectionDivider :divider-classes="getDividerClasses" />

      <!-- 英文内容区域 -->
      <ContentSection
        language="english"
        title="English"
        :paragraphs="getEnglishParagraphs"
        :current-segment="currentSegment"
        :font-size="configEnglishFontSize"
                :section-classes="getEnglishSectionClasses"
        :header-classes="getEnglishHeaderClasses"
        :is-fullscreen="isEnglishFullscreen"
        :is-waiting-for-service="isWaitingForService"
        :last-current-en="lastCurrentEn"
        :auto-scroll="configAutoScroll"
        @report="openReport"
        @fullscreen="toggleEnglishFullscreen"
        @font-size-change="configEnglishFontSize = $event"
        @scroll="onEnglishScroll"
        @toggle-auto-scroll="toggleAutoScroll"
        @scroll-to-bottom="scrollToBottom"
      />
      </div>

      <!-- 报错反馈弹窗 -->
      <CommonReportDialog :visible="showReport" @close="showReport = false" />
    </div>
  </ClientOnly>
</template>

<style>
body {
  font-family: Arial, sans-serif;
  line-height: 1.6;
  margin: 0;
  padding: 0;
}

body.dark {
  background-color: #000;
  color: #ddd;
}
</style>

<style scoped>
/* CSS 变量定义 */
:root {
  --primary-color: #00adb5;
  --primary-hover: #00c4cc;
  --primary-rgb: 0, 173, 181;
  --text-color: #2c3e50;
  --bg-color: #ffffff;
  --border-color: #e2e8f0;
  --hover-bg: #f7fafc;
}

.dark {
  --text-color: #e2e8f0;
  --bg-color: #1a202c;
  --border-color: #4a5568;
  --hover-bg: #2d3748;
}

/* 布局样式 */
.optimized-layout {
  height: 100dvh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.content-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* 隐藏状态 */
.chinese-section.hidden,
.english-section.hidden {
  display: none;
}

.divider {
  height: 2px;
  background: linear-gradient(90deg, #00adb5, #f38181);
  margin: 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  flex-shrink: 0;
}

.divider.hidden {
  opacity: 0;
  pointer-events: none;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 20px;
  background-color: #f8f9fa;
  border-bottom: 1px solid #dee2e6;
  flex-shrink: 0;
  user-select: none;
  height: 42px;
  box-sizing: border-box;
}

.section-header.disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.section-header.fullscreen {
  opacity: 1;
  cursor: pointer;
}


.fullscreen-btn {
  display: flex;
  align-items: center;
}

.section-header h3 {
  margin: 0;
  color: var(--text-color);
  font-size: 1rem;
  font-weight: 600;
}

.section-header h3.chinese-title {
  color: #00adb5;
}

.section-header h3.clickable {
  cursor: pointer;
  transition: opacity 0.2s ease;
  user-select: none;
}

.section-header h3.clickable:hover {
  opacity: 0.8;
}

.section-header.disabled h3.clickable {
  cursor: not-allowed;
  opacity: 0.6;
}

/* 字号控制样式 */
.font-size-controls {
  display: flex;
  gap: 4px;
  align-items: center;
  padding: 8px;
  margin: -8px;
}

.content-area {
  display: flex;
  flex-direction: column;
  flex: 1;
  padding: 15px 20px;
  overflow-y: hidden;
  overflow-x: hidden;
}

/* 文章样式 */
.chinese-article,
.english-article {
  width: 800px;
  margin: 0 auto;
}

.chinese-article.flex-article,
.english-article.flex-article {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  margin-bottom: 1em;
}

.article-paragraph {
  margin-bottom: 1.5em;
  position: relative;
  transition: all 0.3s ease;
}

.article-paragraph.optimizing {
  opacity: 0.8;
}

.article-paragraph.translating {
  opacity: 0.9;
}

.paragraph-content {
  color: #2c3e50;
  font-size: var(--chinese-font-size, 1.3rem);
  line-height: 1.8;
  text-align: justify;
  text-indent: 2em;
  font-weight: 400;
  margin-bottom: 0.5em;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* 英文段落样式 */
.english-content .paragraph-content {
  color: var(--text-color);
  text-align: left;
  font-size: var(--english-font-size, 1.2rem);
  text-indent: 0;
}

/* 英文样式 */
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

/* 翻译内容特殊动画 */
.translation-content {
  display: inline;
  animation: fadeInUp 1.5s ease-out forwards;
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

/* 欢迎信息样式更新 */
.welcome-message {
  text-align: center;
  color: #666;
  font-style: italic;
  padding: 40px 20px;
  font-size: var(--chinese-font-size, 1.3rem);
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}

.english-content .welcome-message {
  font-size: var(--english-font-size, 1.2rem);
}

/* 暗色主题 */
.dark .paragraph-content {
  color: #e0e0e0;
}

.dark .section-header {
  background-color: #1a1a1a;
  border-bottom-color: #333;
}

.dark .section-header h3 {
  color: #00adb5;
}

.dark .chinese-section {
  border-bottom-color: #333;
}

.dark .divider {
  background: linear-gradient(90deg, #00adb5, #f38181);
}

.dark .english-segment {
  color: #e0e0e0;
}

.dark .english-segment.pending {
  color: #666;
}

.dark .welcome-message {
  color: #999;
}

/* 全局滚动条样式 */
:deep(.article-display),
:deep(.menu-panel) {
  scrollbar-width: thin;
}

:deep(.article-display::-webkit-scrollbar),
:deep(.menu-panel::-webkit-scrollbar) {
  width: 4px;
}

:deep(.article-display::-webkit-scrollbar-track),
:deep(.menu-panel::-webkit-scrollbar-track) {
  background: transparent;
}

:deep(.article-display::-webkit-scrollbar-thumb),
:deep(.menu-panel::-webkit-scrollbar-thumb) {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 2px;
}

:deep(.article-display::-webkit-scrollbar-thumb:hover),
:deep(.menu-panel::-webkit-scrollbar-thumb:hover) {
  background: rgba(0, 0, 0, 0.3);
}

:deep(.dark .article-display::-webkit-scrollbar-thumb),
:deep(.dark .menu-panel::-webkit-scrollbar-thumb) {
  background: rgba(255, 255, 255, 0.2);
}

:deep(.dark .article-display::-webkit-scrollbar-thumb:hover),
:deep(.dark .menu-panel::-webkit-scrollbar-thumb:hover) {
  background: rgba(255, 255, 255, 0.3);
}

/* 菜单样式 */
.menu-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.3);
  z-index: 1000;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 80px;
  backdrop-filter: blur(4px);
  animation: fadeIn 0.3s ease;
}

.dark .menu-overlay {
  background-color: rgba(0, 0, 0, 0.6);
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.menu-panel {
  background-color: var(--bg-color, #ffffff);
  border-radius: 16px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.25);
  width: 92%;
  max-width: 420px;
  max-height: 85vh;
  overflow-y: auto;
  animation: slideUp 0.3s ease;
  border: 1px solid var(--border-color, #e2e8f0);
}

@keyframes slideUp {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.menu-content {
  padding: 0;
}

.menu-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px;
  border-bottom: 1px solid var(--border-color, #e2e8f0);
  background: linear-gradient(135deg, var(--bg-color, #ffffff) 0%, rgba(var(--primary-rgb, 0, 173, 181), 0.05) 100%);
  border-radius: 16px 16px 0 0;
}

.menu-header h3 {
  margin: 0;
  font-size: 1.3rem;
  color: var(--text-color, #2c3e50);
  font-weight: 600;
  letter-spacing: 0.5px;
}

.menu-close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--text-color, #2c3e50);
  padding: 8px;
  border-radius: 8px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
}

.menu-close-btn:hover {
  background-color: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  transform: rotate(90deg);
}

.menu-section {
  padding: 24px;
  border-bottom: 1px solid var(--border-color, #e2e8f0);
}

.menu-section:last-child {
  border-bottom: none;
}

.menu-section h4 {
  margin: 0 0 20px 0;
  font-size: 1rem;
  color: var(--primary-color, #00adb5);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-size: 0.85rem;
  opacity: 0.8;
}

.menu-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 12px 0;
  gap: 12px;
}

.menu-item:last-child {
  margin-bottom: 0;
}

.item-info {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
  min-width: fit-content;
}

.item-icon {
  font-size: 1.2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  color: var(--primary-color, #00adb5);
}

/* 版本切换特殊样式 */
.version-section {
  border-bottom: 2px solid var(--border-color, #e2e8f0);
  background: linear-gradient(135deg, rgba(var(--primary-rgb, 0, 173, 181), 0.03) 0%, transparent 100%);
  padding: 16px 0;
  margin-bottom: 0;
}

.version-item {
  justify-content: flex-end;
  padding: 0;
  gap: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
  border-radius: 8px;
}

.version-item:hover {
  background-color: rgba(var(--primary-rgb, 0, 173, 181), 0.06);
  transform: translateX(-2px);
}

.version-item .item-label {
  color: var(--text-color, #2c3e50);
  font-size: 0.95rem;
  font-weight: 500;
  opacity: 0.75;
  transition: all 0.2s ease;
}

.version-item:hover .item-label {
  opacity: 1;
  color: var(--primary-color, #00adb5);
}

.version-icon {
  opacity: 0.6;
  transition: all 0.2s ease;
  color: #666;
}

.version-item:hover .version-icon {
  opacity: 1;
  transform: scale(1.1);
  color: var(--primary-color, #00adb5);
}

.theme-item {
  cursor: pointer;
  transition: all 0.2s ease;
  border-radius: 8px;
}

.theme-item:hover {
  background-color: var(--hover-bg, #f7fafc);
}

.theme-item:hover .theme-toggle-btn {
  transform: scale(1.1);
}

.theme-item:active .theme-toggle-btn {
  transform: scale(0.95);
}

.theme-toggle-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  pointer-events: none;
  padding: 8px;
  border-radius: 8px;
  transition: transform 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 44px;
  min-height: 44px;
}

/* 段落长度选项样式 */
.paragraph-length-item {
  padding-top: 0;
  margin-top: -10px;
}

.paragraph-length-options {
  display: flex;
  gap: 6px;
  width: 100%;
  justify-content: space-between;
}

.length-option {
  flex: 1;
  background: transparent;
  border: 2px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85rem;
  color: var(--text-color, #2c3e50);
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
  text-align: center;
  position: relative;
  overflow: hidden;
}

.length-option:hover {
  background-color: var(--hover-bg, #f7fafc);
  border-color: var(--primary-color, #00adb5);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0, 173, 181, 0.1);
}

.length-option.active {
  background: linear-gradient(135deg, var(--primary-color, #00adb5) 0%, var(--primary-hover, #00c4cc) 100%);
  border-color: var(--primary-color, #00adb5);
  color: white;
  box-shadow: 0 4px 16px rgba(0, 173, 181, 0.4);
  transform: translateY(-1px);
  font-weight: 600;
}

.length-option.active::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s ease;
}

.length-option.active:hover::before {
  left: 100%;
}

.length-option:active {
  transform: translateY(0);
}

/* Toggle Switch 样式 */
.switch {
  position: relative;
  display: inline-block;
  width: 52px;
  height: 28px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #cbd5e1;
  transition: all 0.3s ease;
  border-radius: 28px;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
}

.slider:before {
  position: absolute;
  content: "";
  height: 20px;
  width: 20px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  transition: all 0.3s ease;
  border-radius: 50%;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}

input:checked + .slider {
  background-color: var(--primary-color, #00adb5);
  box-shadow: 0 0 16px rgba(0, 173, 181, 0.4);
}

input:checked + .slider:before {
  transform: translateX(24px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.slider:hover {
  background-color: #94a3b8;
}

input:checked + .slider:hover {
  background-color: var(--primary-hover, #00c4cc);
}

.menu-actions {
  padding: 24px;
  background: linear-gradient(135deg, rgba(var(--primary-rgb, 0, 173, 181), 0.02) 0%, transparent 100%);
}

.action-btn {
  width: 100%;
  background: linear-gradient(135deg, var(--primary-color, #00adb5) 0%, var(--primary-hover, #00c4cc) 100%);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 16px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  box-shadow: 0 4px 16px rgba(0, 173, 181, 0.3);
}

.action-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 173, 181, 0.4);
}

.action-btn:active {
  transform: translateY(0);
  box-shadow: 0 2px 8px rgba(0, 173, 181, 0.3);
}

.action-btn.secondary {
  background: transparent;
  color: var(--text-color, #2c3e50);
  border: 2px solid var(--border-color, #e2e8f0);
  box-shadow: none;
}

.action-btn.secondary:hover {
  background-color: var(--hover-bg, #f7fafc);
  border-color: var(--primary-color, #00adb5);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 173, 181, 0.2);
}

.action-btn.secondary:active {
  transform: translateY(0);
  box-shadow: none;
}


/* 字号按钮样式 */
.font-size-btn {
  background: none;
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 6px;
  width: 40px;
  height: 32px;
  font-size: 1rem;
  color: var(--text-color, #2c3e50);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  padding: 0;
  line-height: 1;
}

.font-size-btn:hover {
  background-color: var(--hover-bg, #f7fafc);
}

.font-size-btn:active {
  border-color: var(--primary-color, #00adb5);
  color: var(--primary-color, #00adb5);
}

/* 当前输入样式 */
.current-input {
  opacity: 0.6;
  font-style: italic;
  border-left: 3px solid #00adb5;
  padding-left: 15px;
  flex-shrink: 0;
  height: var(--chinese-input-height, 1.8em);
  overflow: hidden;
}

.current-input-content {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
  direction: rtl;
  text-align: left;
  text-indent: 0.8em !important;
  color: #2c3e50;
  font-size: var(--chinese-font-size, 1.3rem);
  line-height: 1.8;
}

.truncated-text::after {
  content: "\200E";
}

.blinking-cursor {
  position: relative;
  top: -1px;
  animation: 1s blink step-end infinite;
  margin-right: 2px;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}

.english-input {
  height: var(--english-input-height, 1.8em);
}

.english-input .current-input-content {
  color: var(--text-color, #2c3e50);
  font-size: var(--english-font-size, 1.2rem);
}

.english-content .truncated-text {
  padding-right: 8px;
}

/* 文本段动画 */
.segments-container {
  display: inline;
}

.text-segment {
  display: inline;
  padding: 2px 4px;
  margin: -2px -4px;
  border-radius: 4px;
}

/* 中文文本样式 */
.optimized-text {
  color: #0088cc;
  font-weight: 500;
}

.unoptimized-text {
  color: #888;
}

/* 英文文本样式 */
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

.translating-dots {
  animation: dots 1.5s infinite;
}

@keyframes dots {
  0%, 20% { opacity: 0; }
  50% { opacity: 1; }
  80%, 100% { opacity: 0; }
}

.translation-content {
  display: inline;
  animation: fadeInUp 1.5s ease-out forwards;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .content-area {
    padding: 15px 20px;
  }

  .section-header {
    padding: 10px 20px;
  }

  .section-header h3 {
    font-size: 1.1rem;
  }

  .paragraph-content {
    font-size: var(--chinese-font-size, 1.3rem);
    text-indent: 1em;
  }

  .english-content .paragraph-content {
    font-size: var(--english-font-size, 1.2rem);
    text-indent: 0;
  }

  .chinese-article,
  .english-article {
    width: 100%;
  }

  .article-paragraph {
    margin-bottom: 1.2em;
  }

  .welcome-message {
    padding: 30px 15px;
  }
}

@media (max-width: 480px) {
  .content-area {
    padding: 12px 15px;
  }

  .paragraph-content {
    font-size: var(--chinese-font-size, 1.3rem);
    text-indent: 1em;
    line-height: 1.6;
  }

  .english-content .paragraph-content {
    font-size: var(--english-font-size, 1.2rem);
  }

  .section-header {
    padding: 8px 15px;
  }

  .section-header h3 {
    font-size: 1rem;
  }

  .article-paragraph {
    margin-bottom: 1em;
  }

  .welcome-message {
    padding: 20px 10px;
  }
}
</style>