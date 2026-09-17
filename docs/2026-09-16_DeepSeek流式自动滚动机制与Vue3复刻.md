# DeepSeek Web 流式回复自动滚动机制分析与 Vue3 复刻

> 定稿日期：2026-09-16
> 分析对象：chat.deepseek.com 前端（React + 自研 ds-design-system，bundle `main.39d5f46438.js`）
> 用途：mryk24 聊天/流式输出场景的"自动跟随滚动"实现参考

## TL;DR

DeepSeek 流式回复时的"自动滚到底、用户上滚就停、滚回底部又恢复"，**不是**靠监听滚轮方向，也**不区分**事件是不是用户触发的（不检查 `isTrusted`）。它是一个纯几何判定的状态机：

1. 维护一个布尔标记 `isFollowing`（初始 `true`）；
2. 每次 `scroll` 事件重算：**距底部 < 1px 即视为"跟随中"，否则停止跟随**；
3. 流式内容每次增长时，若 `isFollowing` 为真，把 `scrollTop` 直接写到底（`behavior: "instant"`，不用平滑滚动）。

程序自动滚动写完 `scrollTop` 必然贴底 → 判定仍是"跟随中"，状态自洽；用户滚轮上移 → 不贴底 → 停止；用户滚回最底 → 恢复。**全程不需要任何事件来源判断。**

---

## 一、机制拆解（源码级）

以下代码来自对 `https://fe-static.deepseek.com/chat/static/main.39d5f46438.js` 的反混淆还原（变量名已改为可读形式）。

### 1.1 核心状态机：isFollowing + 贴底判定

跟随逻辑挂在一个自定义 hook 里，核心是三个 ref 和四个方法：

```js
// 还原后的核心 hook（React 版，仅示意）
function useFollowScroll({ virtualListRef, sessionId }) {
  const isFollowing = useRef(true);        // ← 跟随开关，唯一状态
  const stoppedRecord = useRef(null);      // clamp 停止记录（见 1.3）

  // ① 每次 scroll 事件都会执行：贴底判定就是开关本身
  const onUserScroll = () => {
    const el = virtualListRef.current?.listElRef.current;
    if (!el) return;
    isFollowing.current =
      el.scrollTop + el.offsetHeight >= el.scrollHeight - 1;   // ← 阈值 1px
  };

  // ② 流式内容增长时执行：只有跟随中才滚
  const requestFollowScroll = () => {
    if (!isFollowing.current) return;                    // ← 用户上滚后在这里短路
    virtualListRef.current.submitScrollAdjustingTask(followTask);
    followTask();
  };

  // ③ 强制回底（点⬇按钮/切会话），无视 isFollowing
  const forceScrollToBottom = () => {
    virtualListRef.current.scrollTo({ position: "bottom" });
  };
}
```

关键点：**程序滚动与用户滚动走同一个 `scroll` 事件、同一套贴底判定**。自动滚动把视口钉在底部，每次触发判定结果都是 `true`，所以不会自己把自己关掉；只有用户真的滚离底部，判定才变 `false`。这就是"无需区分 isTrusted"的原因。

### 1.2 流式驱动链路

```
SSE delta 首包到达
  → sessionStore.requireScrollToBottom(sessionId)
  → store.scrollToBottomTrigger++            （自增计数器）
  → 组件 useLayoutEffect 监听计数器变化
  → requestFollowScroll()
  → 立即执行 followTask + 注册"粘性滚动意图"

之后每个 token 让内容变高：
  → 虚拟列表 ResizeObserver 重测量
  → 重放粘性意图槽位里的任务（eT.current）
  → followTask() 再次执行，视口继续贴底
```

两个值得借鉴的工程细节：

- **粘性意图槽位**（`eT.current = { type: "bottom" | "custom", task }`）：任务不是一次性的，注册后每次列表重测量都会重放，直到被新意图替换。这解决了"滚动时机永远比内容增长晚一步"的问题。
- **计数器触发**：store 里用 `scrollToBottomTrigger` / `forceScrollToBottomTrigger` 两个自增计数器做事件总线，组件层 `watch` 计数器即可，避免到处传回调。

### 1.3 长回答的 clamp：钉住开头就停

跟随任务里有一段精妙的限制逻辑 —— **回答超过一屏后不再无限贴底**：

```js
// followTask 内部（还原）
const bottomTarget = el.scrollHeight - el.clientHeight;   // 最大 scrollTop

// 实测回答正文起点（.ds-assistant-message-main-content）
// 相对列表顶部的距离（即把它钉到视口顶部所需的滚动量）
const bodyTopOffset = mainContent.getBoundingClientRect().top
                    - el.getBoundingClientRect().top;
const target = currentScrollTop + bodyTopOffset;

if (target >= bottomTarget - 1) {
  // 内容还不足一屏 → 直接贴底，继续跟随
  el.scrollTo({ top: bottomTarget, behavior: "instant" });
} else {
  // 回答已超一屏 → 把回答开头钉在视口顶部，然后停止跟随（每条消息只触发一次）
  el.scrollTo({ top: target, behavior: "instant" });
  stoppedRecord.current = { sessionId, messageId };
  isFollowing.current = false;
}
```

这解释了实际体验：**长回答生成到一屏后，页面停在"回答开头位于视口顶部"，不再继续往下拽**，此时⬇按钮出现。

### 1.4 回到底部按钮与"自愈"

- 按钮显隐独立判定：**距底 ≥ 52px 显示，< 52px 隐藏**（注意与跟随开关的 1px 阈值不同：距底 1~52px 之间时"不跟随但按钮也不显示"）。
- 点按钮 → `forceScrollToBottom` → 注册 `{type: "bottom"}` 粘性意图 + 滚到底；滚到底后 `scroll` 事件触发贴底判定，`isFollowing` 自动恢复 `true` —— **按钮点击不需要手动重置跟随状态，靠几何判定自愈**。
- 按钮仅在流式进行中（`isSending`）渲染；闲时会话上滚不出现按钮（实测确认）。

### 1.5 阈值汇总

| 行为 | 阈值 | 说明 |
|---|---|---|
| 跟随开关（停止/恢复） | 距底 **1px** | scroll 事件里判定 |
| ⬇按钮显隐 | 距底 **52px** | 独立于跟随开关 |
| 长回答停止跟随 | 回答正文起点越过视口顶 | 每条消息一次性 |
| 滚动方式 | `behavior: "instant"` | 高频瞬移，非平滑滚动 |

---

## 二、Vue3 复刻

机制与框架无关。DeepSeek 里的 React 角色 → Vue3 对应关系：

| DeepSeek (React) | 作用 | Vue3 对应 |
|---|---|---|
| `isFollowing` ref | 跟随开关 | 普通变量（模板不渲染它，无需 `ref`） |
| 容器 `onScroll` | 开关判定 + 按钮显隐 | `@scroll` / `addEventListener`，完全一样 |
| store 计数器 + `useLayoutEffect` | 通知"内容长了" | **`ResizeObserver`**（推荐）或 `watch` + `nextTick` |
| 虚拟列表粘性任务重放 | 每个 token 跟一次 | 非虚拟列表不需要，RO 天然每次高度变化触发 |
| `scrollTo({behavior:"instant"})` | 瞬移贴底 | `el.scrollTop = el.scrollHeight` |

### 2.1 composable（可直接放进 mryk24 的 composables/）

```ts
// composables/useFollowScroll.ts
import { ref, onMounted, onBeforeUnmount, type Ref } from 'vue'

export function useFollowScroll(
  container: Ref<HTMLElement | null>,   // 滚动容器
  content: Ref<HTMLElement | null>,     // 内容元素（被流式输出撑高的那个）
) {
  const showJumpBtn = ref(false)        // ⬇按钮显隐（模板响应）
  let isFollowing = true                // 跟随开关（非响应式）
  let ro: ResizeObserver | null = null

  const distToBottom = () => {
    const el = container.value
    return el ? el.scrollHeight - el.scrollTop - el.clientHeight : Infinity
  }

  // ① 唯一的状态开关：贴底几何判定
  const onScroll = () => {
    const dist = distToBottom()
    isFollowing = dist < 1              // 阈值 1px
    showJumpBtn.value = dist >= 52      // 阈值 52px
  }

  const followToBottom = () => {
    const el = container.value
    if (el) el.scrollTop = el.scrollHeight   // instant，直接赋值
  }

  // ② 点⬇按钮：滚到底后 scroll 事件自动把 isFollowing 置回 true，无需手动重置
  const jumpToBottom = followToBottom

  // ③ 外部在流式会话开始时调用：从头跟随（如刚发送新消息）
  const resetFollow = () => { isFollowing = true; showJumpBtn.value = false }

  onMounted(() => {
    container.value?.addEventListener('scroll', onScroll, { passive: true })
    ro = new ResizeObserver(() => {
      if (isFollowing) followToBottom()
      // 可选：在这里加 DeepSeek 的长回答 clamp——
      // 测 content 中当前回答的 top，越过视口顶就 isFollowing = false（记 messageId 防重入）
    })
    if (content.value) ro.observe(content.value)
  })

  onBeforeUnmount(() => {
    container.value?.removeEventListener('scroll', onScroll)
    ro?.disconnect()
  })

  return { showJumpBtn, jumpToBottom, resetFollow }
}
```

### 2.2 页面中使用

```vue
<script setup lang="ts">
const listEl = ref<HTMLElement | null>(null)
const contentEl = ref<HTMLElement | null>(null)
const { showJumpBtn, jumpToBottom, resetFollow } = useFollowScroll(listEl, contentEl)

// 发送新消息时重置跟随
async function onSend(text: string) {
  resetFollow()
  // ...发送逻辑
}
</script>

<template>
  <div ref="listEl" class="chat-list">
    <div ref="contentEl">
      <ChatMessage v-for="m in messages" :key="m.id" :msg="m" />
    </div>
  </div>
  <button v-show="showJumpBtn" class="jump-btn" @click="jumpToBottom">⬇</button>
</template>

<style scoped>
.chat-list { overflow-y: auto; height: 100%; }
.jump-btn { position: absolute; right: 24px; bottom: 120px; }
</style>
```

---

## 三、注意事项与坑

1. **别用 `watch` 直接驱动滚动**：watch 回调执行时 DOM 可能还没更新，`scrollHeight` 是旧值。必须 `await nextTick()`。用 `ResizeObserver` 则无此时序问题（回调触发时布局已完成）——**推荐 RO**。
2. **跟随开关用普通变量而非 `ref`**：`isFollowing` 在高频 scroll/RO 回调里读写，包成 `ref` 会引入不必要的响应式开销；模板只依赖 `showJumpBtn`。
3. **虚拟列表场景**（virtua / vue-virtual-scroller）：简单赋值不够——虚拟列表测量是异步的，需要 DeepSeek 那套"粘性意图"：把"滚到底"作为持久任务注册，每次测量后重放，而不是只滚一次。
4. **滚轮事件不需要监听**：容易想当然地去做"wheel 向上就停"，但那要额外处理触摸板惯性、滚动条拖拽、键盘 PgUp 等入口。贴底几何判定天然覆盖所有滚动方式。
5. **发送新消息要显式恢复跟随**：用户上滚阅读旧消息后直接发送新消息，此时 `isFollowing` 是 `false`，新回复不会跟随——DeepSeek 的处理是新消息/切会话走 `forceScrollToBottomTrigger` 强制回底。对应上面 `resetFollow()`。
6. **`behavior: "smooth"` 不要用**：平滑滚动的动画期间再写入新目标会互相打架，高频流式场景必须用瞬移（直接赋值或 `behavior: "instant"`），靠高频调用制造"连续跟随"的观感。

---

## 附：分析证据

- 分析日期：2026-09-16，bundle：`main.39d5f46438.js`（chat.deepseek.com 生产版本，后续版本行号可能漂移但机制稳定）
- 关键类名：滚动容器 `ds-scroll-area`，内部列表 `ds-virtual-list`（transform 虚拟化），回答正文锚点 `.ds-assistant-message-main-content`
- store 字段：session 级 `scrollToBottomTrigger` / `forceScrollToBottomTrigger` 计数器
- 实测验证：滚动容器 React props 挂载 `onScroll` + `onWheel`（onWheel 仅用于侧边栏加载更多/图片缩放等，主状态机不依赖）；⬇按钮仅在流式期间渲染
