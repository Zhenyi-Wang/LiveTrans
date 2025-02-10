<script setup>
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
const configShowText = useStorage("config-show-text", true);
const toggleShowText = useToggle(configShowText);

const configShowTextOpti = useStorage("config-show-text-opti", true);
const toggleShowTextOpti = useToggle(configShowTextOpti);

const configShowTextEn = useStorage("config-show-text-en", true);
const toggleShowTextEn = useToggle(configShowTextEn);

// 滚动控制
const isAutoScroll = ref(true);
const lastScrollTop = ref(0);
const scrollTimer = ref(null);
const scrollStartPosition = ref(0);
const lastScrollHeight = ref(0);
const { y } = useWindowScroll({ behavior: "smooth" });

onMounted(() => {
  // 在组件挂载后初始化滚动位置
  lastScrollTop.value = window.scrollY;
  window.addEventListener('scroll', onScroll);
  connectWS();
});

const onScroll = () => {
  const currentScrollTop = window.scrollY;
  console.log('滚动事件：', {
    当前位置: currentScrollTop,
    上次位置: lastScrollTop.value,
    开始位置: scrollStartPosition.value,
    是否自动模式: isAutoScroll.value,
    滚动方向: currentScrollTop < lastScrollTop.value ? '向上' : '向下'
  });
  
  if (isAutoScroll.value) {
    // 检测到向上滚动
    if (currentScrollTop < lastScrollTop.value - 10) {  // 添加一个阈值，避免微小的滚动
      console.log('检测到向上滚动，准备切换到手动模式');
      // 如果已经有一个定时器在运行，清除它
      if (scrollTimer.value) {
        clearTimeout(scrollTimer.value);
      }
      
      // 记录开始检测时的位置
      scrollStartPosition.value = lastScrollTop.value;
      
      // 设置一个新的定时器，如果在 100ms 内持续向上滚动，才切换到手动模式
      scrollTimer.value = setTimeout(() => {
        const finalScrollTop = window.scrollY;
        console.log('定时器触发，检查最终位置：', {
          最终位置: finalScrollTop,
          开始位置: scrollStartPosition.value,
          差值: scrollStartPosition.value - finalScrollTop
        });
        
        if (finalScrollTop < scrollStartPosition.value - 20) {
          console.log('确认向上滚动，切换到手动模式');
          isAutoScroll.value = false;
        } else {
          console.log('滚动不满足条件，保持自动模式');
        }
        scrollTimer.value = null;
      }, 100);
    }
  } else {
    // 在手动模式下，检测是否滚动到底部
    const scrollHeight = document.documentElement.scrollHeight;
    const windowHeight = window.innerHeight;
    const scrolledToBottom = currentScrollTop + windowHeight >= scrollHeight - 10; // 添加10px的容差
    
    if (scrolledToBottom) {
      console.log('用户已滚动到底部，切换到自动模式');
      isAutoScroll.value = true;
    }
  }
  
  lastScrollTop.value = currentScrollTop;
  lastScrollHeight.value = document.documentElement.scrollHeight;
};

const scrollToBottom = () => {
  console.log('执行滚动到底部');
  const targetScrollTop = document.documentElement.scrollHeight - window.innerHeight;
  window.scrollTo({
    top: targetScrollTop,
    behavior: 'smooth'
  });
  
  if (!isAutoScroll.value) {
    console.log('切换到自动模式：用户点击按钮');
    isAutoScroll.value = true;
  }
};

onUnmounted(() => {
  window.removeEventListener('scroll', onScroll);
  if (scrollTimer.value) {
    clearTimeout(scrollTimer.value);
  }
});

// 数据相关
const ws = ref(null);
const currentSegment = ref({ text: "" });
const lastCurrentEn = ref("");
const confirmedSegments = reactive({ value: [] });
const wsConnected = ref(false);

const isWaitingForService = computed(() => {
  return (
    currentSegment.value.text === "" && confirmedSegments.value.length === 0
  );
});

// 输入框高度跟随
const input = ref(null);
const inputPadding = ref(null);
const inputHeight = ref(0);

const updateInputHeight = () => {
  // console.log("updateInputHeight")
  if (input.value) {
    inputHeight.value = input.value.clientHeight;
    // console.log("inputHeight", inputHeight.value)
  }
};

watch(currentSegment, () => {
  updateInputHeight();
});

// 监听内容变化
watchThrottled(
  [currentSegment, confirmedSegments],
  () => {
    if (isAutoScroll.value) {
      console.log('内容更新，执行自动滚动');
      nextTick(() => {
        // 清除可能存在的定时器
        if (scrollTimer.value) {
          clearTimeout(scrollTimer.value);
          scrollTimer.value = null;
        }
        scrollToBottom();
      });
    } else {
      console.log('内容更新，但处于手动模式，不滚动');
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
    // console.log(event.data)
    let data = JSON.parse(event.data);

    if (data.init) {
      currentSegment.value = data.init.current;
      confirmedSegments.value = data.init.confirmed;
    }
    if (data.current) {
      currentSegment.value = data.current;
    }
    if (data.current_en) {
      lastCurrentEn.value = data.current_en.en_text;
    }
    if (data.confirmed) {
      // console.log("data.confirmed", data.confirmed)
      confirmedSegments.value.push(data.confirmed);
      // console.log("confirmedSegments", confirmedSegments.value)
    }
    if (data.update) {
      // console.log("data.update", data.update)
      const index = confirmedSegments.value.findLastIndex((s) => {
        return s.id === data.update.id;
      });
      // console.log('index', index)
      if (index >= 0) {
        confirmedSegments.value[index] = data.update;
      }
    }
  };
};
</script>

<template>
  <ClientOnly>
    <h1 v-if="false">
      Welcome to Haining Xiashi Christ Church. We're delighted to have you here
      on Sundays!
    </h1>
    <div v-else>
      <header>
        <h2>Live Translation (Beta)</h2>
        <div v-show="!showMenu" class="menu-open-btn" @click="toggleMenu()">
          ⚙️
        </div>
        <div v-show="showMenu" class="menu">
          <div class="menu-close-btn" @click="toggleMenu()">X</div>
          <ul>
            <li @click="toggleDark()">
              <span v-if="!isDark"> 🌞 Light Mode | 亮色主题</span>
              <span v-else>🌙 Dark Mode | 暗色主题</span>
            </li>
            <li @click="toggleShowText()">
              <span v-if="!configShowText">☐ </span>
              <span v-else>✔️ </span><span>Show Chinese 1 | 显示中文1</span>
            </li>
            <li @click="toggleShowTextOpti()">
              <span v-if="!configShowTextOpti">☐ </span>
              <span v-else>✔️ </span><span>Show Chinese 2 | 显示中文2 </span>
            </li>
            <li @click="toggleShowTextEn()">
              <span v-if="!configShowTextEn">☐ </span>
              <span v-else>✔️ </span><span>Show English | 显示英文</span>
            </li>
          </ul>
        </div>
      </header>
      <div id="app">
        <div v-if="isWaitingForService">
          Welcome to Haining Xiashi Christ Church!😊 We're delighted to have you
          here. Please be patient and wait while we begin...
        </div>
        <div
          v-for="segment of confirmedSegments.value"
          class="confirmed"
          style="margin-bottom: 10px"
        >
          <!--transition name="slide"-->
          <div class="seg">
            <span v-show="configShowText" class="seg-text"
              >{{ segment.text }}<br
            /></span>
            <span v-show="configShowTextOpti" class="seg-opti-text">{{
              segment.opti_text || "..."
            }}</span>
          </div>
          <!--/transition-->
          <div v-show="configShowTextEn" class="seg-en-text">
            <!--transition name="slide"-->
            <div v-if="segment.en_text">{{ segment.en_text }}</div>
            <div v-else>
              <div class="loader"></div>
            </div>
            <!--/transition-->
          </div>
        </div>
        <div
          ref="input-padding"
          style="margin-top: 20px"
          :style="{ height: inputHeight + 'px' }"
        ></div>
        <div ref="input" class="input-container">
          <!-- 🗣 -->
          🔊 <span>{{ currentSegment.text }}</span
          ><span v-show="!isWaitingForService" class="blinking-cursor"> |</span>
          <br />
          <span v-if="configShowTextEn && lastCurrentEn">{{
            lastCurrentEn
          }}</span>
          <span class="blinking-cursor" v-else></span>

          <div v-if="wsConnected" class="connection-dot"></div>
          <div v-else class="connection-dot disconnected"></div>
        </div>
      </div>
      <!-- 滚动到底部按钮 -->
      <div v-show="!isAutoScroll" class="scroll-bottom-btn" @click="scrollToBottom">
        ⬇️
      </div>
    </div>
  </ClientOnly>
</template>

<style>
body {
  font-family: Arial, sans-serif;
  line-height: 1.6;
  margin: 0;
  padding: 10px;
}

body.dark {
  background-color: #000;
  color: #ddd;
}
</style>

<style scoped>
.scheme {
  color: #222831;
  color: #393e46;
  color: #00adb5;
  color: #eeeeee;
  color: #f38181;
  color: #fce38a;
  color: #eaffd0;
  color: #95e1d3;
}

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

.dark .menu li {
  color: #fff;
}

.dark .menu {
  background-color: #000;
}

.input-container {
  background-color: #fff;
  position: fixed;
  z-index: 1;
  bottom: 0;
  left: 0;
  right: 0;
  color: #00adb5;
  border: 1px solid #00adb5;
  padding: 10px;
  padding-right: 30px;
  border-radius: 4px;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
}

.dark .input-container {
  background-color: #000;
  color: #00adb5;
  box-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
}

.seg-text {
  color: #00adb5;
}

.seg-opti-text {
  color: #222831;
}

.seg-en-text {
  color: #f38181;
}

.dark .seg-text {
  color: #00adb5;
}

.dark .seg-opti-text {
  color: #f38181;
}

.dark .seg-en-text {
  color: #eeeeee;
}

.connection-dot {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  right: 10px;
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: #00ff55;
}

.connection-dot.disconnected {
  background-color: #ff0000;
}

/* 定义一个闪烁的光标 */
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

/* 载入圆圈 */
.loader {
  display: inline-block;
  border: 3px solid #555;
  /* Light grey */
  border-top: 3px solid #0d2b40;
  /* Blue */
  border-radius: 50%;
  width: 0.6rem;
  height: 0.6rem;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% {
    transform: rotate(20deg);
  }

  30% {
    transform: rotate(220deg);
    opacity: 0.5;
  }

  /* 50% {
    transform: rotate(180deg);
    opacity: 1;
  }
  70% {
    transform: rotate(200deg);
    opacity: 0.5;
  } */
  100% {
    transform: rotate(380deg);
  }
}

/* 定义一个slide动画 */
.slide-enter-active {
  transition: all 0.5s ease;
  transform-origin: bottom;
}

.slide-leave-active {
  transition: all 0.5s ease;
  transform-origin: top;
  height: 0;
}

.slide-leave-to {
  opacity: 0;
  height: 0;
  transform: rotateX(90deg);
  /* transform: translateY(-100%); */
}

.slide-enter-from {
  transform: rotateX(-90deg);
}

.slide-enter-to,
.slide-leave-from {
  opacity: 100;
  /* height: 0; */
  /* transform: translateX(0); */
}

.el-container {
  height: 100vh;
}

.scroll-bottom-btn {
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: rgba(255, 255, 255, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  transition: all 0.3s ease;
  z-index: 1000;
}

.scroll-bottom-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.dark .scroll-bottom-btn {
  background-color: rgba(50, 50, 50, 0.8);
}
</style>
