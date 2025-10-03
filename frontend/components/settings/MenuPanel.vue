<template>
  <div class="menu-overlay" @click="$emit('close')">
    <div class="menu-panel" @click.stop>
      <div class="menu-content">
        <!-- 版本切换 -->
        <div class="menu-section version-section">
          <div class="menu-item version-item" @click="$emit('goToOriginal')">
            <span class="item-label">回到旧版界面 | Classic UI</span>
            <div class="item-icon version-icon">
              <FontAwesomeIcon icon="history" />
            </div>
          </div>
        </div>

        <div class="menu-header">
          <h3>设置 | Settings</h3>
          <button class="menu-close-btn" @click="$emit('close')">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>

        <!-- 显示设置 -->
        <div class="menu-section">
          <h4>显示设置 | Display</h4>
          <div class="menu-item theme-item" @click="toggleDark">
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
            <BaseToggle
              :model-value="configAutoScroll"
              @update:model-value="toggleAutoScroll"
            />
          </div>

          <div class="menu-item">
            <div class="item-info">
              <span class="item-icon"><FontAwesomeIcon icon="sync" /></span>
              <span class="item-label">联动滚动 | Sync Scroll</span>
            </div>
            <BaseToggle
              :model-value="configSyncScroll"
              @update:model-value="toggleSyncScroll"
            />
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
                @click="configParagraphLength = 100"
              >
                100
              </button>
              <button
                class="length-option"
                :class="{ 'active': configParagraphLength === 150 }"
                @click="configParagraphLength = 150"
              >
                150
              </button>
              <button
                class="length-option"
                :class="{ 'active': configParagraphLength === 200 }"
                @click="configParagraphLength = 200"
              >
                200
              </button>
              <button
                class="length-option"
                :class="{ 'active': configParagraphLength === 300 }"
                @click="configParagraphLength = 300"
              >
                300
              </button>
            </div>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div class="menu-section">
          <div class="menu-actions">
            <BaseButton variant="secondary" class="action-btn" @click="$emit('close')">
              关闭 | Close
            </BaseButton>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
// 引入配置
const {
  isDark,
  toggleDark,
  configAutoScroll,
  toggleAutoScroll,
  configSyncScroll,
  toggleSyncScroll,
  configParagraphLength
} = useAppConfig()

defineEmits(['close', 'goToOriginal'])
</script>

<style scoped>
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

.menu-item span {
  color: var(--text-color, #2c3e50);
  font-size: 0.95rem;
  font-weight: 500;
  letter-spacing: 0.2px;
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

.menu-actions {
  padding: 24px;
  background: linear-gradient(135deg, rgba(var(--primary-rgb, 0, 173, 181), 0.02) 0%, transparent 100%);
}

.action-btn {
  width: 100%;
  padding: 16px;
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

</style>