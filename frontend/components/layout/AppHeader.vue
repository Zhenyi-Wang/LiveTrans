<template>
  <header>
    <div class="header-content">
      <div class="header-left">
        <CommonConnectionStatus :connected="wsConnected" />
        <h2>Live Translation | 实时翻译</h2>
      </div>
      <div class="header-right">
        <div v-show="!showMenu" class="menu-btn" @click="$emit('toggleMenu')">
          <FontAwesomeIcon icon="cog" />
        </div>
      </div>
    </div>

    <!-- 菜单面板 -->
    <div v-show="showMenu" class="menu-overlay" @click="$emit('toggleMenu')">
      <div class="menu-panel" @click.stop>
        <div class="menu-content">
          <!-- 版本切换 -->
          <div class="menu-section version-section">
            <div class="menu-item version-item" @click="goToOriginalVersion">
              <span class="item-label">回到旧版界面 | Classic UI</span>
              <div class="item-icon version-icon">
                <FontAwesomeIcon icon="history" />
              </div>
            </div>
          </div>

          <div class="menu-header">
            <h3>设置 | Settings</h3>
            <button class="menu-close-btn" @click="$emit('toggleMenu')">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>

          <!-- 显示设置 -->
          <div class="menu-section">
            <h4>显示设置 | Display</h4>
            <div class="menu-item theme-item" @click="$emit('toggleDark')">
              <div class="item-info">
                <span class="item-icon"><FontAwesomeIcon :icon="isDark ? 'moon' : 'sun'" /></span>
                <span class="item-label">主题 | Theme</span>
              </div>
              <div class="theme-toggle-btn">
                <FontAwesomeIcon :icon="isDark ? 'moon' : 'sun'" />
              </div>
            </div>

            <div class="menu-item">
              <div class="item-info">
                <span class="item-icon"><FontAwesomeIcon icon="arrow-down" /></span>
                <span class="item-label">自动滚动 | Auto Scroll</span>
              </div>
              <label class="switch">
                <input type="checkbox" :checked="configAutoScroll" @change="$emit('toggleAutoScroll'); configAutoScroll && $emit('scrollToBottom')">
                <span class="slider"></span>
              </label>
            </div>

            <div class="menu-item">
              <div class="item-info">
                <span class="item-icon"><FontAwesomeIcon icon="sync" /></span>
                <span class="item-label">联动滚动 | Sync Scroll</span>
              </div>
              <label class="switch">
                <input type="checkbox" :checked="configSyncScroll" @change="$emit('toggleSyncScroll')">
                <span class="slider"></span>
              </label>
            </div>

            <div class="menu-item">
              <div class="item-info">
                <span class="item-icon"><FontAwesomeIcon icon="paragraph" /></span>
                <span class="item-label">段落长度 | Paragraph Length</span>
              </div>
            </div>
            <div class="menu-item paragraph-length-item">
              <div class="paragraph-length-options">
                <button
                  class="length-option"
                  :class="{ 'active': configParagraphLength === 100 }"
                  @click="$emit('update:configParagraphLength', 100)"
                >
                  100
                </button>
                <button
                  class="length-option"
                  :class="{ 'active': configParagraphLength === 200 }"
                  @click="$emit('update:configParagraphLength', 200)"
                >
                  200
                </button>
                <button
                  class="length-option"
                  :class="{ 'active': configParagraphLength === 300 }"
                  @click="$emit('update:configParagraphLength', 300)"
                >
                  300
                </button>
                <button
                  class="length-option"
                  :class="{ 'active': configParagraphLength === 400 }"
                  @click="$emit('update:configParagraphLength', 400)"
                >
                  400
                </button>
              </div>
            </div>

            <div class="menu-item">
              <div class="item-info">
                <span class="item-icon"><FontAwesomeIcon icon="font" /></span>
                <span class="item-label">中文字号 | Chinese Font Size</span>
              </div>
              <div class="font-size-controls">
                <button class="font-size-btn" @click="$emit('update:configChineseFontSize', Math.max(0.8, configChineseFontSize - 0.1))">
                  <FontAwesomeIcon icon="minus" />
                </button>
                <span class="font-size-value">{{ configChineseFontSize.toFixed(1) }}</span>
                <button class="font-size-btn" @click="$emit('update:configChineseFontSize', Math.min(2.0, configChineseFontSize + 0.1))">
                  <FontAwesomeIcon icon="plus" />
                </button>
              </div>
            </div>

            <div class="menu-item">
              <div class="item-info">
                <span class="item-icon"><FontAwesomeIcon icon="font" /></span>
                <span class="item-label">英文字号 | English Font Size</span>
              </div>
              <div class="font-size-controls">
                <button class="font-size-btn" @click="$emit('update:configEnglishFontSize', Math.max(0.8, configEnglishFontSize - 0.1))">
                  <FontAwesomeIcon icon="minus" />
                </button>
                <span class="font-size-value">{{ configEnglishFontSize.toFixed(1) }}</span>
                <button class="font-size-btn" @click="$emit('update:configEnglishFontSize', Math.min(2.0, configEnglishFontSize + 0.1))">
                  <FontAwesomeIcon icon="plus" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup>
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import {
  faCog,
  faMoon,
  faSun,
  faArrowDown,
  faSync,
  faParagraph,
  faMinus,
  faPlus,
  faHistory,
  faFont,
  faEye
} from '@fortawesome/free-solid-svg-icons'
import { library } from '@fortawesome/fontawesome-svg-core'

// 添加图标到库
library.add(
  faCog,
  faMoon,
  faSun,
  faArrowDown,
  faSync,
  faParagraph,
  faMinus,
  faPlus,
  faHistory,
  faFont,
  faEye
)

defineProps({
  wsConnected: {
    type: Boolean,
    required: true
  },
  showMenu: {
    type: Boolean,
    required: true
  },
  isDark: {
    type: Boolean,
    required: true
  },
  configAutoScroll: {
    type: Boolean,
    required: true
  },
  configSyncScroll: {
    type: Boolean,
    required: true
  },
  configParagraphLength: {
    type: Number,
    required: true
  },
  configChineseFontSize: {
    type: Number,
    required: true
  },
  configEnglishFontSize: {
    type: Number,
    required: true
  },
  configShowText: {
    type: Boolean,
    required: true
  },
  configShowTextEn: {
    type: Boolean,
    required: true
  }
})

defineEmits([
  'toggleMenu',
  'toggleDark',
  'toggleAutoScroll',
  'toggleSyncScroll',
  'scrollToBottom',
  'update:configParagraphLength',
  'update:configChineseFontSize',
  'update:configEnglishFontSize',
  'update:configShowText',
  'update:configShowTextEn'
])

const goToOriginalVersion = () => {
  window.location.href = '/index-original'
}
</script>

<style scoped>
/* Header样式 */
.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0px 20px;
  background-color: #f8f9fa;
  border-bottom: 1px solid #dee2e6;
  flex-shrink: 0;
  min-height: 40px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

header h2 {
  margin: 0;
  font-size: 1rem;
  color: var(--text-color, #2c3e50);
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.menu-btn {
  background: none;
  border: none;
  font-size: 1rem;
  cursor: pointer;
  color: var(--text-color, #2c3e50);
  padding: 6px;
  border-radius: 6px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 32px;
  min-height: 32px;
  opacity: 0.7;
}

.menu-btn:hover {
  background-color: var(--hover-bg, #f7fafc);
  transform: scale(1.1);
  opacity: 1;
}

.menu-btn:active {
  transform: scale(0.95);
}

/* 菜单面板样式 */
.menu-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100dvh;
  background-color: rgba(0, 0, 0, 0.5);
  z-index: 1000;
  display: flex;
  align-items: flex-start;
  justify-content: flex-end;
  padding: 0;
  box-sizing: border-box;
}

.menu-panel {
  width: 400px;
  max-width: 90vw;
  height: 100dvh;
  background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
  backdrop-filter: blur(10px);
  box-shadow: -10px 0 30px rgba(0, 0, 0, 0.1);
  border-radius: 20px 0 0 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-right: none;
  overflow: hidden;
  position: relative;
  animation: slideInRight 0.3s ease-out;
  max-height: 100dvh;
  display: flex;
  flex-direction: column;
}

@keyframes slideInRight {
  from {
    transform: translateX(100%);
  }
  to {
    transform: translateX(0);
  }
}

.menu-content {
  height: 100%;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 24px;
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
  cursor: pointer;
  padding: 8px;
  border-radius: 8px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-color, #2c3e50);
}

.menu-close-btn:hover {
  background-color: var(--hover-bg, #f7fafc);
  transform: scale(1.1);
}

.menu-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.menu-section h4 {
  margin: 0 0 8px 0;
  font-size: 0.9rem;
  font-weight: 600;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding-bottom: 4px;
  border-bottom: 1px solid #e2e8f0;
}

.menu-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background-color: #ffffff;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  transition: all 0.3s ease;
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.menu-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  border-color: var(--primary-color, #00adb5);
}

.item-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.item-icon {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--primary-color, #00adb5);
  font-size: 1rem;
}

.item-label {
  font-size: 0.95rem;
  font-weight: 500;
  color: #2d3748;
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

/* 开关样式 */
.switch {
  position: relative;
  display: inline-block;
  width: 50px;
  height: 26px;
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
  background-color: #ccc;
  transition: .4s;
  border-radius: 34px;
}

.slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  transition: .4s;
  border-radius: 50%;
}

input:checked + .slider {
  background-color: var(--primary-color, #00adb5);
}

input:checked + .slider:before {
  transform: translateX(24px);
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
}

/* 字号控制样式 */
.font-size-controls {
  display: flex;
  align-items: center;
  gap: 12px;
}

.font-size-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 50%;
  background: linear-gradient(135deg, #f7fafc 0%, #e2e8f0 100%);
  color: var(--primary-color, #00adb5);
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.font-size-btn:hover {
  transform: scale(1.1);
  box-shadow: 0 4px 8px rgba(0, 173, 181, 0.3);
  background: linear-gradient(135deg, var(--primary-color, #00adb5) 0%, var(--primary-hover, #00c4cc) 100%);
  color: white;
}

.font-size-btn:active {
  transform: scale(0.95);
}

.font-size-value {
  min-width: 30px;
  text-align: center;
  font-weight: 600;
  color: var(--primary-color, #00adb5);
  font-size: 0.9rem;
}

/* 深色主题样式 */
.dark .header-content {
  background-color: #1a1a1a;
  border-bottom-color: #333;
}

.dark header h2 {
  color: #e0e0e0;
}

.dark .menu-btn {
  color: #cccccc;
}

.dark .menu-btn:hover {
  background-color: #404040;
}

.dark .menu-panel {
  background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
  border-color: rgba(255, 255, 255, 0.1);
}

.dark .menu-item {
  background-color: #2d2d2d;
  border-color: #404040;
}

.dark .menu-item:hover {
  border-color: var(--primary-color, #00adb5);
}

.dark .item-label {
  color: #e0e0e0;
}

.dark .menu-section h4 {
  color: #9ca3af;
  border-bottom-color: #404040;
}

.dark .version-item .item-label {
  color: #e0e0e0;
}

.dark .length-option {
  border-color: #404040;
  color: #e0e0e0;
}

.dark .length-option:hover {
  background-color: #404040;
  border-color: var(--primary-color, #00adb5);
}

.dark .menu-header {
  background: linear-gradient(135deg, #1a1a1a 0%, rgba(0, 173, 181, 0.05) 100%);
  border-bottom-color: #404040;
}

.dark .menu-header h3 {
  color: #e0e0e0;
}

.dark .menu-close-btn {
  color: #e0e0e0;
}

.dark .menu-close-btn:hover {
  background-color: #404040;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .header-content {
    padding: 6px 15px;
    min-height: 36px;
  }

  header h2 {
    font-size: 0.9rem;
  }

  .menu-btn {
    min-width: 28px;
    min-height: 28px;
    font-size: 0.9rem;
    padding: 4px;
  }

  .menu-panel {
    width: 80vw;
  }
}
</style>