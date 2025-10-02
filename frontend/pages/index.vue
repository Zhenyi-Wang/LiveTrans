<script setup>
// 引入Font Awesome
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import {
  faCog,
  faFileAlt,
  faMoon,
  faSun,
  faArrowDown,
  faSync,
  faParagraph,
  faExpand,
  faCompress,
  faMinus,
  faPlus,
  faCircle,
  faWifi,
  faSpinner,
  faHistory,
  faExchangeAlt,
  faUndo,
  faStepBackward,
  faReply
} from '@fortawesome/free-solid-svg-icons'
import { library } from '@fortawesome/fontawesome-svg-core'

// 添加图标到库
library.add(
  faCog,
  faFileAlt,
  faMoon,
  faSun,
  faArrowDown,
  faSync,
  faParagraph,
  faExpand,
  faCompress,
  faMinus,
  faPlus,
  faCircle,
  faWifi,
  faSpinner,
  faHistory,
  faExchangeAlt,
  faUndo,
  faStepBackward,
  faReply
)

const languages = usePreferredLanguages();

const isSunday = new Date().getDay() == 0;

// 菜单
const showMenu = ref(false);
const toggleMenu = useToggle(showMenu);

// 主题切换
const isDark = useDark({
  selector: "body",
  attribute: "class",
  valueDark: "dark",
  valueLight: "",
});

const toggleDark = useToggle(isDark);
import { useStorage } from "@vueuse/core";

// 配置
const configAutoScroll = useStorage("config-auto-scroll", true);
const toggleAutoScroll = useToggle(configAutoScroll);

const configShowText = useStorage("config-show-text", true);
const toggleShowText = useToggle(configShowText);

const configShowTextOpti = useStorage("config-show-text-opti", true);
const toggleShowTextOpti = useToggle(configShowTextOpti);

const configShowTextEn = useStorage("config-show-text-en", true);
const toggleShowTextEn = useToggle(configShowTextEn);

const configSyncScroll = useStorage("config-sync-scroll", true);
const toggleSyncScroll = useToggle(configSyncScroll);

const configParagraphLength = useStorage("config-paragraph-length", 300);

const configChineseFontSize = useStorage("config-chinese-font-size", 1.3);
const configEnglishFontSize = useStorage("config-english-font-size", 1.2);

// 全屏状态管理
const isChineseFullscreen = ref(false);
const isEnglishFullscreen = ref(false);

const toggleChineseFullscreen = () => {
  if (isChineseFullscreen.value) {
    // 取消全屏
    isChineseFullscreen.value = false;
  } else {
    // 设置中文全屏，同时取消英文全屏
    isChineseFullscreen.value = true;
    isEnglishFullscreen.value = false;
  }
};

const toggleEnglishFullscreen = () => {
  if (isEnglishFullscreen.value) {
    // 取消全屏
    isEnglishFullscreen.value = false;
  } else {
    // 设置英文全屏，同时取消中文全屏
    isEnglishFullscreen.value = true;
    isChineseFullscreen.value = false;
  }
};

// 滚动结束后的最终同步
const scrollEndSyncTimer = ref(null);

const finalizeScrollSync = (sourceElement, targetSelector) => {
  if (scrollEndSyncTimer.value) {
    clearTimeout(scrollEndSyncTimer.value);
  }

  // 延迟一点时间确保滚动完全结束
  scrollEndSyncTimer.value = setTimeout(() => {
    if (configSyncScroll.value) {
      const targetElement = document.querySelector(targetSelector);
      if (targetElement && sourceElement) {
        const sourceScrollHeight = sourceElement.scrollHeight - sourceElement.clientHeight;
        const targetScrollHeight = targetElement.scrollHeight - targetElement.clientHeight;

        if (sourceScrollHeight > 0 && targetScrollHeight > 0) {
          const scrollRatio = sourceElement.scrollTop / sourceScrollHeight;
          const targetScrollTop = scrollRatio * targetScrollHeight;

          // 最终精确同步
          targetElement.scrollTop = targetScrollTop;
        }
      }
    }
  }, 100);
};

// 数据相关
const ws = ref(null);
const currentSegment = ref({ text: "" });
const lastCurrentEn = ref("");
const confirmedSegments = reactive({ value: [] });
const wsConnected = ref(false);

// 段落式排版相关 - 基于confirmedSegments
const currentParagraph = ref("");

// 每段最大字符数 - 使用计算属性响应配置变化
const maxParagraphLength = computed(() => configParagraphLength.value);

const isWaitingForService = computed(() => {
  return (
    currentSegment.value.text === "" &&
    confirmedSegments.value.length === 0 &&
    currentParagraph.value === ""
  );
});

// 第一步：将segments分成段落（统一分段逻辑）
const getSegmentParagraphs = computed(() => {
  const paragraphs = [];
  let currentSegments = [];
  let currentLength = 0;
  const maxLength = maxParagraphLength.value;

  // 处理已确认的segments
  confirmedSegments.value.forEach(segment => {
    if (segment.text) {
      const text = segment.opti_text || segment.text;
      const segmentLength = text.length;

      currentSegments.push(segment);
      currentLength += segmentLength;

      if (currentLength >= maxLength) {
        // 寻找合适的分段点
        let splitIndex = currentSegments.length;
        const sentenceEnds = ['。', '？', '！', '；', '.', '?', '!', ';'];

        // 在当前段落中寻找分段点
        let accumulatedLength = 0;

        for (let i = currentSegments.length - 1; i >= Math.max(0, currentSegments.length - 10); i--) {
          accumulatedLength += (currentSegments[i].opti_text || currentSegments[i].text).length;
          if (currentLength - accumulatedLength <= maxLength - 50) {
            break;
          }

          const segmentText = currentSegments[i].opti_text || currentSegments[i].text;
          for (let j = segmentText.length - 1; j >= Math.max(0, segmentText.length - 50); j--) {
            if (sentenceEnds.includes(segmentText[j])) {
              splitIndex = i;
              break;
            }
          }
          if (splitIndex < currentSegments.length) break;
        }

        // 如果没找到合适的分段点，就在当前segment处分段
        if (splitIndex === currentSegments.length) {
          splitIndex = currentSegments.length - 1;
        }

        // 保存段落
        const paragraphSegments = currentSegments.slice(0, splitIndex);
        paragraphs.push(paragraphSegments);

        // 重置 - 计算剩余segments的长度
        const segmentLength = paragraphSegments.reduce((sum, seg) =>
          sum + (seg.opti_text || seg.text).length, 0);
        currentSegments = currentSegments.slice(splitIndex);
        currentLength -= segmentLength;
      }
    }
  });

  // 处理剩余的segments
  if (currentSegments.length > 0) {
    paragraphs.push(currentSegments);
  }

  return paragraphs;
});

// 第二步：根据分段生成中文段落
const getChineseParagraphs = computed(() => {
  const paragraphs = getSegmentParagraphs.value.map(segmentGroup => {
    let html = "";
    segmentGroup.forEach(seg => {
      const text = seg.opti_text || seg.text;
      if (seg.opti_text && seg.opti_text !== seg.text) {
        html += `<span class="optimized-text">${text}</span>`;
      } else {
        html += text;
      }
    });
    return { html };
  });

  // 添加当前正在输入的段落
  if (currentParagraph.value) {
    paragraphs.push({ html: currentParagraph.value });
  }

  return paragraphs;
});

// 第二步：根据分段生成英文段落
const getEnglishParagraphs = computed(() => {
  const paragraphs = getSegmentParagraphs.value.map(segmentGroup => {
    let enText = "";
    segmentGroup.forEach(seg => {
      if (seg.en_text) {
        enText += (enText ? " " : "") + seg.en_text;
      }
    });
    return enText || null;
  });

  // 添加当前正在输入的段落
  if (currentParagraph.value) {
    paragraphs.push(null);
  }

  return paragraphs;
});


// 联动滚动相关
const isScrolling = ref(false);
const scrollSource = ref(''); // 'chinese' 或 'english'

// 自动滚动 - 分别控制中文和英文区域
const scrollToBottom = () => {
  // 滚动中文区域到底部
  const chineseContent = document.querySelector('.chinese-content');
  if (chineseContent) {
    chineseContent.scrollTop = chineseContent.scrollHeight + 1000;
  }

  // 滚动英文区域到底部
  const englishContent = document.querySelector('.english-content');
  if (englishContent) {
    englishContent.scrollTop = englishContent.scrollHeight + 1000;
  }
};

// 联动滚动函数
const syncScroll = (sourceElement, targetSelector) => {
  if (isScrolling.value) return;

  isScrolling.value = true;
  const targetElement = document.querySelector(targetSelector);

  if (targetElement && sourceElement) {
    // 计算滚动比例
    const sourceScrollHeight = sourceElement.scrollHeight - sourceElement.clientHeight;
    const targetScrollHeight = targetElement.scrollHeight - targetElement.clientHeight;

    if (sourceScrollHeight > 0 && targetScrollHeight > 0) {
      const scrollRatio = sourceElement.scrollTop / sourceScrollHeight;
      const targetScrollTop = scrollRatio * targetScrollHeight;

      // 直接设置滚动位置，不使用平滑滚动（避免冲突）
      targetElement.scrollTop = targetScrollTop;

      // 使用 requestAnimationFrame 确保滚动位置准确
      requestAnimationFrame(() => {
        targetElement.scrollTop = targetScrollTop;
      });
    }
  }

  // 减少阻塞时间，快速重置状态
  setTimeout(() => {
    isScrolling.value = false;
  }, 50);
};

// 节流处理的滚动事件
const onChineseScrollThrottled = (event) => {
  if (!isScrolling.value && configSyncScroll.value) {
    scrollSource.value = 'chinese';
    syncScroll(event.target, '.english-content');
    // 触发最终同步
    finalizeScrollSync(event.target, '.english-content');
  }
};

const onEnglishScrollThrottled = (event) => {
  if (!isScrolling.value && configSyncScroll.value) {
    scrollSource.value = 'english';
    syncScroll(event.target, '.chinese-content');
    // 触发最终同步
    finalizeScrollSync(event.target, '.chinese-content');
  }
};

// 使用节流函数包装滚动事件（减少节流间隔，提高响应性）
const onChineseScroll = useThrottleFn(onChineseScrollThrottled, 16);
const onEnglishScroll = useThrottleFn(onEnglishScrollThrottled, 16);

watchThrottled(
  [currentSegment, confirmedSegments],
  () => {
    if (configAutoScroll.value) {
      scrollToBottom();
    }
  },
  { throttle: 1000 }
);

// 连接相关
const reconnectWS = () => {
  console.log("Reconnecting WebSocket...");
  setTimeout(connectWS, 1000);
};

const connectWS = () => {
  // const isSecure = location.protocol === "https:"
  // const url = (isSecure ? "wss://" : "ws://") + location.host + "/api/ws"

  // ws.value = new WebSocket(url)
  ws.value = new WebSocket("/backend/api/ws");

  console.log("Establishing WebSocket connection...");

  const timeoutId = setTimeout(() => {
    if (ws.value.readyState === WebSocket.CONNECTING) {
      console.log("WebSocket connection timed out, retrying in 3 seconds...");
      ws.value.close();
    }
  }, 5000);

  // 连接成功处理
  ws.value.onopen = () => {
    console.log("WebSocket connection established");
    wsConnected.value = true;
  };

  // 连接错误处理
  ws.value.onerror = (error) => {
    console.log("WebSocket error observed:", error);
    wsConnected.value = false;
  };

  // 连接关闭处理
  ws.value.onclose = (event) => {
    console.log("WebSocket connection closed", event);
    wsConnected.value = false;
    reconnectWS();
  };

  ws.value.onmessage = (event) => {
    let data = JSON.parse(event.data);
    console.log('WebSocket received:', data);

    if (data.init) {
      currentSegment.value = data.init.current;
      confirmedSegments.value = data.init.confirmed;
    }
    if (data.current) {
      currentSegment.value = data.current;
      // 添加到当前段落缓冲
      currentParagraph.value += data.current.text;
    }
    if (data.current_en) {
      lastCurrentEn.value = data.current_en.en_text;
    }
    if (data.confirmed) {
      confirmedSegments.value.push(data.confirmed);
    }
    if (data.update) {
      console.log('Received update:', data.update);
      // 使用原始页面的逻辑更新confirmedSegments
      const index = confirmedSegments.value.findLastIndex((s) => {
        return s.id === data.update.id;
      });
      console.log('Found segment at index:', index);
      if (index >= 0) {
        confirmedSegments.value[index] = data.update;
        console.log('Updated segment:', confirmedSegments.value[index]);
      }
    }
  };
};

// 跳转到旧版页面
const goToOriginalVersion = () => {
  window.location.href = '/index-original';
};

onMounted(() => {
  connectWS();
});
</script>

<template>
  <ClientOnly>
    <div class="optimized-layout" :style="{
      '--chinese-font-size': configChineseFontSize + 'rem',
      '--english-font-size': configEnglishFontSize + 'rem'
    }">
      <header>
        <div class="header-content">
          <div class="header-left">
            <div class="main-connection-status">
              <div v-if="wsConnected" class="connection-dot connected"></div>
              <div v-else class="connection-dot disconnected"></div>
            </div>
            <h2>Live Translation | 实时翻译</h2>
          </div>
          <div class="header-right">
            <div v-show="!showMenu" class="menu-btn" @click="toggleMenu()">
              <FontAwesomeIcon icon="cog" />
            </div>
          </div>
        </div>
        <div v-show="showMenu" class="menu-overlay" @click="toggleMenu()">
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
              <button class="menu-close-btn" @click="toggleMenu()">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </button>
            </div>

              <!-- 显示设置 -->
              <div class="menu-section">
                <h4>显示设置 | Display</h4>
                <div class="menu-item theme-item" @click="toggleDark()">
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
                    <input type="checkbox" :checked="configAutoScroll" @change="toggleAutoScroll(); configAutoScroll && scrollToBottom()">
                    <span class="slider"></span>
                  </label>
                </div>

                <div class="menu-item">
                  <div class="item-info">
                    <span class="item-icon"><FontAwesomeIcon icon="sync" /></span>
                    <span class="item-label">联动滚动 | Sync Scroll</span>
                  </div>
                  <label class="switch">
                    <input type="checkbox" :checked="configSyncScroll" @change="toggleSyncScroll()">
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
                  <button class="action-btn secondary" @click="toggleMenu()">
                    关闭 | Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
        </header>

      <!-- 上半部分：中文内容 -->
      <div class="chinese-section" :class="{ 'hidden': isEnglishFullscreen, 'fullscreen': isChineseFullscreen }">
        <div class="section-header" :class="{ 'disabled': isEnglishFullscreen, 'fullscreen': isChineseFullscreen }" @click="toggleChineseFullscreen">
          <div class="header-left">
            <div class="fullscreen-btn">
              <button class="font-size-btn" @click.stop="toggleChineseFullscreen">
                <FontAwesomeIcon :icon="isChineseFullscreen ? 'compress' : 'expand'" />
              </button>
            </div>
            <h3 class="chinese-title">中文 | Chinese</h3>
          </div>
          <div class="header-right">
            <div class="font-size-controls">
              <button class="font-size-btn" @click.stop="configChineseFontSize = Math.max(0.8, configChineseFontSize - 0.1)">
                <FontAwesomeIcon icon="minus" />
              </button>
              <button class="font-size-btn" @click.stop="configChineseFontSize = Math.min(2.0, configChineseFontSize + 0.1)">
                <FontAwesomeIcon icon="plus" />
              </button>
            </div>
          </div>
        </div>

        <div class="content-area chinese-content" @scroll="onChineseScroll">
          <div v-if="isWaitingForService" class="welcome-message">
            欢迎来到海宁市硖石基督教堂！😊 我们很高兴您的到来。请耐心等待，我们即将开始...
          </div>

          <!-- 完整的中文段落显示 -->
          <div class="chinese-article">
            <div
              v-for="(paragraph, index) in getChineseParagraphs"
              :key="index"
              class="article-paragraph"
            >
              <div class="paragraph-content" v-html="paragraph.html"></div>
            </div>

            <!-- 当前正在输入的段落 -->
            <div v-if="currentParagraph && configShowText" class="current-paragraph">
              <div class="paragraph-content">
                {{ currentParagraph }}
                <span class="blinking-cursor"> |</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 分隔线 -->
      <div class="divider" :class="{ 'hidden': isChineseFullscreen || isEnglishFullscreen }"></div>

      <!-- 下半部分：英文内容 -->
      <div class="english-section" :class="{ 'hidden': isChineseFullscreen, 'fullscreen': isEnglishFullscreen }">
        <div class="section-header" :class="{ 'disabled': isChineseFullscreen, 'fullscreen': isEnglishFullscreen }" @click="toggleEnglishFullscreen">
          <div class="header-left">
            <div class="fullscreen-btn">
              <button class="font-size-btn" @click.stop="toggleEnglishFullscreen">
                <FontAwesomeIcon :icon="isEnglishFullscreen ? 'compress' : 'expand'" />
              </button>
            </div>
            <h3>English</h3>
          </div>
          <div class="header-right">
            <div class="font-size-controls">
              <button class="font-size-btn" @click.stop="configEnglishFontSize = Math.max(0.8, configEnglishFontSize - 0.1)">
                <FontAwesomeIcon icon="minus" />
              </button>
              <button class="font-size-btn" @click.stop="configEnglishFontSize = Math.min(2.0, configEnglishFontSize + 0.1)">
                <FontAwesomeIcon icon="plus" />
              </button>
            </div>
          </div>
        </div>

        <div class="content-area english-content" @scroll="onEnglishScroll">
          <div v-if="isWaitingForService" class="welcome-message">
            Welcome to Haining Xiashi Christ Church!😊 We're delighted to have you here. Please be patient and wait while we begin...
          </div>

          <!-- 完整的英文段落显示 -->
          <div class="english-article">
            <div
              v-for="(paragraph, index) in getEnglishParagraphs"
              :key="index"
              class="article-paragraph"
              :class="{ 'translating': !paragraph }"
            >
              <div class="paragraph-content english-content">
                <div v-if="paragraph">{{ paragraph }}</div>
                <div v-else class="loading-text">
                  <FontAwesomeIcon icon="spinner" spin />
                  <span>Translating...</span>
                </div>
              </div>
              <!-- 翻译状态指示器 -->
              <div v-if="!paragraph" class="status-indicator">
                <FontAwesomeIcon icon="spinner" spin />
                <span class="status-text">翻译中...</span>
              </div>
            </div>

            <!-- 当前英文翻译 -->
            <div v-if="lastCurrentEn && configShowTextEn" class="current-translation">
              <div class="paragraph-content english-content">
                {{ lastCurrentEn }}
              </div>
            </div>
          </div>
        </div>
      </div>
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

header {
  flex-shrink: 0;
  z-index: 100;
  padding: 0 20px;
  height: 45px;
  background-color: var(--bg-color);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 15px;
  flex: 1;
  min-width: 0; /* 防止flex子项溢出 */
}

.header-right {
  display: flex;
  align-items: center;
  flex-shrink: 0;
}

.chinese-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-bottom: 2px solid #e0e0e0;
  transition: all 0.3s ease;
}

.english-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: all 0.3s ease;
}

/* 全屏状态 */
.chinese-section.fullscreen {
  flex: 1;
  border-bottom: none;
}

.english-section.fullscreen {
  flex: 1;
}

/* 隐藏状态 */
.chinese-section.hidden {
  display: none;
}

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

.section-header {
  cursor: pointer;
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

.fullscreen-btn {
  display: flex;
  align-items: center;
}

header h2 {
  margin: 0;
  font-size: 1.2rem;
  color: var(--text-color);
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.menu-btn {
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: var(--text-color);
  padding: 6px 8px;
  border-radius: 4px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0.7;
}

.menu-btn:hover {
  background-color: var(--hover-bg);
  opacity: 1;
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

/* 字号控制样式 */
.font-size-controls {
  display: flex;
  gap: 4px;
  align-items: center;
  padding: 8px;
  margin: -8px;
}

.font-size-btn {
  background: none;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  width: 40px;
  height: 32px;
  font-size: 1rem;
  color: var(--text-color);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  padding: 0;
  line-height: 1;
}


.font-size-btn:active {
  border-color: var(--primary-color);
  color: var(--primary-color);
}

.content-area {
  flex: 1;
  padding: 15px 20px;
  overflow-y: auto;
  overflow-x: hidden;
}


.segment-container {
  margin-bottom: 15px;
  animation: fadeIn 0.5s ease-in;
}

/* 文章样式 */
.chinese-article,
.english-article {
  max-width: 800px;
  margin: 0 auto;
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
  text-indent: 2em; /* 首行缩进 */
  font-weight: 400;
  margin-bottom: 0.5em;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* 优化文本的颜色 */
.optimized-text {
  color: #0088cc;
  font-weight: 500;
}

.current-paragraph {
  opacity: 0.8;
  font-style: italic;
  border-left: 3px solid #00adb5;
  padding-left: 15px;
  margin-left: 2em;
}

.current-translation {
  opacity: 0.8;
  font-style: italic;
  border-left: 3px solid var(--text-color);
  padding-left: 15px;
  margin-left: 15px;
}

.dark .current-paragraph {
  border-left-color: #00adb5;
}

.dark .current-translation {
  border-left-color: var(--text-color);
}

/* 状态指示器 */
.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  padding: 4px 12px;
  background-color: #f8f9fa;
  border-radius: 12px;
  font-size: 0.85rem;
  color: #666;
  align-self: flex-start;
}

.status-indicator svg {
  color: var(--primary-color);
  animation: spin 1s linear infinite;
}

.status-text {
  font-style: italic;
}

/* 英文段落样式 */
.english-content .paragraph-content {
  color: var(--text-color);
  text-align: left;
  text-indent: 0; /* 英文不缩进 */
  font-size: var(--english-font-size, 1.2rem);
}

/* 加载文本样式 */
.loading-text {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #999;
  font-style: italic;
  opacity: 0.7;
}

.loading-text svg {
  color: var(--primary-color);
  animation: spin 1s linear infinite;
}

/* 暗色主题 */
.dark .paragraph-content {
  color: #e0e0e0;
}

.dark .optimized-text {
  color: #66b3ff;
  font-weight: 500;
}

.dark .english-content .paragraph-content {
  color: var(--text-color);
}

.dark .status-indicator {
  background-color: #2d3748;
  color: #999;
}

.dark .loading-text {
  color: #666;
}

/* 欢迎信息样式更新 */
.welcome-message {
  text-align: center;
  color: #666;
  font-style: italic;
  padding: 40px 20px;
  font-size: 1.1rem;
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}

/* 英文样式 */
.english-segment {
  font-size: 1.1rem;
  line-height: 1.6;
  color: #f38181;
  min-height: 1.6em;
}

.english-segment.pending {
  color: #ccc;
  font-style: italic;
}

.loading-text {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #999;
  font-style: italic;
}


/* 欢迎信息 */
.welcome-message {
  text-align: center;
  color: #666;
  font-style: italic;
  padding: 40px 20px;
  font-size: 1.1rem;
}

/* 菜单样式 */
.menu-open-btn,
.menu-close-btn {
  font-size: 1.5rem;
  position: fixed;
  z-index: 1001;
  top: 2rem;
  right: 2rem;
}

.menu {
  position: fixed;
  z-index: 1000;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  background-color: #fff;
}

.menu ul {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.menu li {
  margin: 10px 0;
  font-size: 1.2rem;
  cursor: pointer;
}

/* 暗色主题 */
.dark .menu li {
  color: #fff;
}

.dark .menu {
  background-color: #000;
}

.dark .section-header {
  background-color: #1a1a1a;
  border-bottom-color: #333;
}

.dark .section-header h3 {
  color: #00adb5;
}

.dark .collapse-icon {
  color: #999;
}

.dark .chinese-section {
  border-bottom-color: #333;
}

.dark .divider {
  background: linear-gradient(90deg, #00adb5, #f38181);
}

.dark .original-text {
  color: #00adb5;
}

.dark .original-text.updated {
  color: #666;
}

.dark .optimized-text {
  color: #e0e0e0;
  background-color: #1a1a1a;
  border-left-color: #00adb5;
}

.dark .english-segment {
  color: #f38181;
}

.dark .english-segment.pending {
  color: #666;
}

.dark .loading-text {
  color: #666;
}

.dark .current-input {
  background-color: #2d3748;
  border-color: #4a5568;
  color: #00adb5;
}

.dark .english-current {
  color: #f38181;
}

.dark .welcome-message {
  color: #999;
}

.dark .both-collapsed-notice {
  color: #999;
}

/* 连接状态点 */
.main-connection-status {
  display: flex;
  align-items: center;
  margin-right: 15px;
}

.connection-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: #00ff55;
  box-shadow: 0 0 8px rgba(0, 255, 85, 0.4);
  transition: all 0.3s ease;
}

.connection-dot.connected {
  background-color: #00ff55;
  box-shadow: 0 0 8px rgba(0, 255, 85, 0.4);
}

.connection-dot.disconnected {
  background-color: #ff4444;
  box-shadow: 0 0 8px rgba(255, 68, 68, 0.4);
}

.connection-dot.disconnected {
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.7;
    transform: scale(1.1);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
}

/* 闪烁光标 */
.blinking-cursor {
  position: relative;
  top: -1px;
  animation: 1s blink step-end infinite;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
}


@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

/* 动画 */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes slideInRight {
  from {
    opacity: 0;
    transform: translateX(-20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}


/* 滚动条样式 */
.content-area {
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 173, 181, 0.3) transparent;
}

.content-area::-webkit-scrollbar {
  width: 8px;
}

.content-area::-webkit-scrollbar-track {
  background: transparent;
}

.content-area::-webkit-scrollbar-thumb {
  background: rgba(0, 173, 181, 0.3);
  border-radius: 4px;
  transition: background 0.2s ease;
}

.content-area::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 173, 181, 0.5);
}

.dark .content-area::-webkit-scrollbar-thumb {
  background: rgba(0, 173, 181, 0.4);
}

.dark .content-area::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 173, 181, 0.6);
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
    text-indent: 1.5em;
  }

  .english-content .paragraph-content {
    font-size: var(--english-font-size, 1.2rem);
  }

  .chinese-article,
  .english-article {
    max-width: 100%;
  }

  .article-paragraph {
    margin-bottom: 1.2em;
  }

  .status-indicator {
    font-size: 0.8rem;
    padding: 3px 10px;
  }

  .welcome-message {
    font-size: 1rem;
    padding: 30px 15px;
  }
}

@media (max-width: 480px) {
  .content-area {
    padding: 12px 15px;
  }

  .paragraph-content {
    font-size: var(--chinese-font-size, 1.3rem);
    text-indent: 1.2em;
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
    font-size: 0.95rem;
    padding: 20px 10px;
  }

  .status-indicator {
    font-size: 0.75rem;
    padding: 2px 8px;
  }
}

/* 菜单样式 */
.menu-toggle {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  width: 24px;
  height: 18px;
  cursor: pointer;
  z-index: 1001;
  transition: all 0.3s ease;
}

.menu-toggle:hover div {
  background-color: var(--primary-color);
}

.menu-toggle div {
  width: 100%;
  height: 2px;
  background-color: var(--text-color);
  transition: all 0.3s ease;
  border-radius: 1px;
}

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
  background-color: var(--bg-color);
  border-radius: 16px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.25);
  width: 92%;
  max-width: 420px;
  max-height: 85vh;
  overflow-y: auto;
  animation: slideUp 0.3s ease;
  border: 1px solid var(--border-color);
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

.menu-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px;
  border-bottom: 1px solid var(--border-color);
  background: linear-gradient(135deg, var(--bg-color) 0%, rgba(var(--primary-rgb, 0, 173, 181), 0.05) 100%);
  border-radius: 16px 16px 0 0;
}

.menu-header h3 {
  margin: 0;
  font-size: 1.3rem;
  color: var(--text-color);
  font-weight: 600;
  letter-spacing: 0.5px;
}

.menu-close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--text-color);
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
  border-bottom: 1px solid var(--border-color);
}

.menu-section:last-child {
  border-bottom: none;
}

.menu-section h4 {
  margin: 0 0 20px 0;
  font-size: 1rem;
  color: var(--primary-color);
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
  color: var(--text-color);
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
  color: var(--primary-color);
}

/* 版本切换特殊样式 */
.version-section {
  border-bottom: 2px solid var(--border-color);
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
  color: var(--text-color);
  font-size: 0.95rem;
  font-weight: 500;
  opacity: 0.75;
  transition: all 0.2s ease;
}

.version-item:hover .item-label {
  opacity: 1;
  color: var(--primary-color);
}

.version-icon {
  opacity: 0.6;
  transition: all 0.2s ease;
  color: #666;
}

.version-item:hover .version-icon {
  opacity: 1;
  transform: scale(1.1);
  color: var(--primary-color);
}

.theme-btn {
  background: none;
  border: 2px solid var(--border-color);
  border-radius: 12px;
  padding: 10px 16px;
  cursor: pointer;
  font-size: 1.3rem;
  transition: all 0.3s ease;
  background-color: var(--bg-color);
  min-width: 50px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.theme-btn:hover {
  background-color: var(--hover-bg);
  border-color: var(--primary-color);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 173, 181, 0.3);
}

.theme-btn:active {
  transform: translateY(0);
}

.theme-item {
  cursor: pointer;
  transition: all 0.2s ease;
  border-radius: 8px;
}

.theme-item:hover {
  background-color: var(--hover-bg);
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
  pointer-events: none; /* 让点击事件穿透到父元素 */
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
  border: 2px solid var(--border-color);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 0.85rem;
  color: var(--text-color);
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
  text-align: center;
  position: relative;
  overflow: hidden;
}

.length-option:hover {
  background-color: var(--hover-bg);
  border-color: var(--primary-color);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0, 173, 181, 0.1);
}

.length-option.active {
  background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%);
  border-color: var(--primary-color);
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
.toggle-switch {
  position: relative;
  display: inline-block;
  width: 52px;
  height: 28px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
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

.toggle-slider:before {
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

input:checked + .toggle-slider {
  background-color: var(--primary-color);
  box-shadow: 0 0 16px rgba(0, 173, 181, 0.4);
}

input:checked + .toggle-slider:before {
  transform: translateX(24px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.toggle-slider:hover {
  background-color: #94a3b8;
}

input:checked + .toggle-slider:hover {
  background-color: #00c4cc;
}

.menu-actions {
  padding: 24px;
  background: linear-gradient(135deg, rgba(var(--primary-rgb, 0, 173, 181), 0.02) 0%, transparent 100%);
}

.action-btn {
  width: 100%;
  background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%);
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
  color: var(--text-color);
  border: 2px solid var(--border-color);
  box-shadow: none;
}

.action-btn.secondary:hover {
  background-color: var(--hover-bg);
  border-color: var(--primary-color);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 173, 181, 0.2);
}

.action-btn.secondary:active {
  transform: translateY(0);
  box-shadow: none;
}

/* 滚动条样式 */
.menu-panel {
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 173, 181, 0.3) transparent;
}

.menu-panel::-webkit-scrollbar {
  width: 8px;
}

.menu-panel::-webkit-scrollbar-track {
  background: transparent;
}

.menu-panel::-webkit-scrollbar-thumb {
  background: rgba(0, 173, 181, 0.3);
  border-radius: 4px;
  transition: background 0.2s ease;
}

.menu-panel::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 173, 181, 0.5);
}

.dark .menu-panel::-webkit-scrollbar-thumb {
  background: rgba(0, 173, 181, 0.4);
}

.dark .menu-panel::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 173, 181, 0.6);
}
</style>
