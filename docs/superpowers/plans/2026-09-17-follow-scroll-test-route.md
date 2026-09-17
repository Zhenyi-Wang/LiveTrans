# 自动跟随滚动 v2（/st 对比测试页）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec v2 实现全自动跟随滚动（取消手动开关），以**纯新增副本**方式挂在不暴露的测试路由 `/st` 上，与线上版并存供对比测试。

**Architecture:** 新 composable `useFollowScroll`（scroll 事件三分类协议 + 追赶式缓动 + 联动 rAF 合并器）；三个组件副本（ParagraphDisplayV2/SectionHeaderV2/ContentSectionV2）承载结构改动；`pages/st.vue` 复用现有数据层（useWebSocket/useParagraphLogic/useFullscreen/useAppConfig）与 AppHeader 组装。**现有文件零改动**，两版路由互不影响。

**Tech Stack:** Nuxt 3 / Vue 3 Composition API / @vueuse/core（现有依赖，无新增）。

**Spec:** `docs/2026-09-16_自动跟随滚动重构spec_取消手动开关.md`（v2 定稿，含复审修订）

## Global Constraints

- **禁止修改任何现有文件**——只允许 Create 新文件（对比测试的前提）。
- **禁止 git commit**（going-out 约定，完成后等用户审核提交）。
- 前端包管理用 **yarn**，命令在 `frontend/` 目录执行；**禁止引入新依赖**。
- 阈值常量（spec 2.2/2.5 定值）：`EPS=1`、`DRIFT=8`、`BOTTOM=1`、`JUMP_AT=52`、`TAU=300`、`MAX_DT=50`（单位 px / ms）。
- CSS 不变量（spec 3.2）：滚动容器 `scroll-behavior: auto` 保持、`overflow-anchor: none` 新增（仅 V2 副本的 scoped style）。
- 项目无前端单测基建（package.json 无 test script），验证 = `yarn build` 构建通过 + spec 第五节手动验收清单；不引入测试框架（范围外）。
- 所有新组件无 `<script lang="ts">`，沿用项目 JS 组件惯例；composable 用 TS（与 `useScrollSync.ts` 一致）。

---

### Task 1: `useFollowScroll` composable（核心机制）

**Files:**
- Create: `frontend/composables/useFollowScroll.ts`

**Interfaces:**
- Consumes: `configSyncScroll: Ref<boolean>`（来自 `useAppConfig()`）。
- Produces: `showJumpCn: Ref<boolean>`、`showJumpEn: Ref<boolean>`、`jumpToBottom(key: 'chinese' | 'english'): void`。内部依赖 DOM 结构 `.chinese-article.article-display` / `.english-article.article-display` 及其直接子层 `.article-body`（Task 2 产出）。

- [ ] **Step 1: 写入完整实现**

```ts
// composables/useFollowScroll.ts
// 自动跟随滚动 v2 —— spec: docs/2026-09-16_自动跟随滚动重构spec_取消手动开关.md
// 核心: scroll 事件三分类(布局指纹/程序写入/外部位移) + 追赶式缓动 + 联动 rAF 合并器
import { ref, onMounted, onBeforeUnmount, type Ref } from 'vue'

// ===== spec 阈值常量 =====
const EPS = 1        // 程序写入自身事件的噪声容差(px)
const DRIFT = 8      // 向上累计介入阈值(px)
const BOTTOM = 1     // 贴底判定(px)
const JUMP_AT = 52   // 跳底按钮显隐阈值(px)
const TAU = 300      // 追赶动画时间常数(ms)
const MAX_DT = 50    // 单帧 dt 上限(ms),防后台标签页回前台 k≈1 瞬移

type LaneKey = 'chinese' | 'english'

interface LaneState {
  key: LaneKey
  el: HTMLElement
  isFollowing: boolean
  lastWritten: number                    // 最后已知 scrollTop(程序写入读回值/外部位移新基线)
  lastGeom: { h: number; ch: number }    // 布局指纹
  userDrift: number                      // 向上累计外部位移(≤0)
  running: boolean
  raf: number | null
  lastTs: number
  roLastH: number                        // 内容 RO 自持的上次观测高度(display:none 判定用)
  handleScroll: () => void
  handleWheel: (e: WheelEvent) => void
}

export function useFollowScroll(configSyncScroll: Ref<boolean>) {
  const showJumpCn = ref(false)
  const showJumpEn = ref(false)

  const lanes: Partial<Record<LaneKey, LaneState>> = {}
  let bodyRo: ResizeObserver | null = null
  let boxRo: ResizeObserver | null = null
  let syncRaf: number | null = null
  let pendingSync: LaneState | null = null
  let mq: MediaQueryList | null = null
  let mqHandler: ((e: MediaQueryListEvent) => void) | null = null
  let reducedMotion = false

  const other = (lane: LaneState) => lanes[lane.key === 'chinese' ? 'english' : 'chinese']
  const showJumpOf = (key: LaneKey) => (key === 'chinese' ? showJumpCn : showJumpEn)
  const maxTopOf = (lane: LaneState) => lane.el.scrollHeight - lane.el.clientHeight

  function programmaticScrollTo(lane: LaneState, top: number) {
    lane.el.scrollTop = top
    lane.lastWritten = lane.el.scrollTop   // 读回:吸收钳制与舍入,随后的自身 scroll 事件走②分支
  }

  function updateJumpBtn(lane: LaneState, dist: number) {
    showJumpOf(lane.key).value = !lane.isFollowing && dist >= JUMP_AT
  }

  // ===== 2.2 scroll 事件三分类协议 =====
  function onScrollLane(lane: LaneState) {
    const el = lane.el
    const h = el.scrollHeight
    const ch = el.clientHeight
    const geomChanged = h !== lane.lastGeom.h || ch !== lane.lastGeom.ch
    lane.lastGeom = { h, ch }
    const maxTop = h - ch
    const top = Math.min(Math.max(el.scrollTop, 0), maxTop)   // clamp 防 overscroll 橡皮筋
    const dist = maxTop - top

    // ① 布局事件(内容收缩/视口变化/钳制/锚定):只刷基线与按钮,绝不改跟随状态
    if (geomChanged) {
      lane.lastWritten = top
      updateJumpBtn(lane, dist)
      return
    }

    const external = top - lane.lastWritten
    // ② 程序写入自身触发的事件:不作联动源、不改状态
    if (Math.abs(external) <= EPS) {
      updateJumpBtn(lane, dist)
      return
    }
    // ③ 外部位移(用户):唯一能停止/恢复跟随的来源
    lane.lastWritten = top
    if (external < 0) {
      lane.userDrift += external
      if (lane.isFollowing && -lane.userDrift > DRIFT) stopFollow(lane)
    } else {
      if (!lane.isFollowing && dist < BOTTOM) resumeFollow(lane)   // 主动滚到底 → 自愈
    }
    updateJumpBtn(lane, dist)
    scheduleSyncIfEligible(lane)
  }

  // wheel 向上仅取消动画(消除拉拽感),是否进入阅读态由位移确认决定
  function onWheelUp(lane: LaneState, e: WheelEvent) {
    if (e.deltaY < 0) cancelLoop(lane)
  }

  function resumeFollow(lane: LaneState) {
    lane.isFollowing = true
    lane.userDrift = 0
    lane.lastWritten = Math.min(Math.max(lane.el.scrollTop, 0), maxTopOf(lane))
    ensureLoop(lane)
    updateJumpBtn(lane, maxTopOf(lane) - lane.lastWritten)
  }

  function stopFollowLocal(lane: LaneState) {
    if (!lane.isFollowing) return
    lane.isFollowing = false
    cancelLoop(lane)
    lane.userDrift = 0
    updateJumpBtn(lane, maxTopOf(lane) - lane.el.scrollTop)
  }

  // 联动范围在此实施:联动开停两栏,联动关只停源栏(spec 2.1)
  function stopFollow(source: LaneState) {
    stopFollowLocal(source)
    if (configSyncScroll.value) {
      const peer = other(source)
      if (peer) stopFollowLocal(peer)
    }
  }

  // ===== 2.5 追赶式缓动 =====
  function ensureLoop(lane: LaneState) {
    if (lane.running || !lane.isFollowing) return
    lane.running = true
    lane.lastTs = performance.now()
    lane.raf = requestAnimationFrame(ts => followLoop(lane, ts))
  }

  function followLoop(lane: LaneState, ts: number) {
    const dt = Math.min(ts - lane.lastTs, MAX_DT)
    lane.lastTs = ts
    const el = lane.el
    const target = maxTopOf(lane)              // 每帧重算,内容增长自动追
    const dist = target - el.scrollTop
    if (dist < BOTTOM) {
      programmaticScrollTo(lane, target)       // 精确贴底后停帧
      lane.userDrift = 0                       // 贴底即漂移清零(防低幅误触跨时间累计)
      lane.running = false
      lane.raf = null
      return
    }
    if (reducedMotion) {
      programmaticScrollTo(lane, target)       // 独立分支,不参与除法
    } else {
      programmaticScrollTo(lane, el.scrollTop + dist * (1 - Math.exp(-dt / TAU)))
    }
    lane.raf = requestAnimationFrame(t2 => followLoop(lane, t2))
  }

  function cancelLoop(lane: LaneState) {
    if (lane.raf !== null) {
      cancelAnimationFrame(lane.raf)
      lane.raf = null
    }
    lane.running = false
  }

  // ===== 2.7 联动:单一 rAF 合并器,仅双方阅读态生效 =====
  function scheduleSyncIfEligible(source: LaneState) {
    if (!configSyncScroll.value || source.isFollowing) return
    const peer = other(source)
    if (!peer || peer.isFollowing) return
    pendingSync = source
    if (syncRaf === null) syncRaf = requestAnimationFrame(applySync)
  }

  function applySync() {
    syncRaf = null
    const src = pendingSync
    pendingSync = null
    if (!src || !configSyncScroll.value || src.isFollowing) return
    const peer = other(src)
    if (!peer || peer.isFollowing) return
    const srcMax = maxTopOf(src)
    const peerMax = maxTopOf(peer)
    if (srcMax > 0 && peerMax > 0) {
      programmaticScrollTo(peer, (Math.min(Math.max(src.el.scrollTop, 0), srcMax) / srcMax) * peerMax)
    }
  }

  // ===== 2.6 内容 RO:布局刷新 + display:none 恢复特例 =====
  function onBodyRoCheck(lane: LaneState) {
    const h = lane.el.scrollHeight
    const prev = lane.roLastH
    lane.roLastH = h
    if (prev === 0 && h === 0) return          // 隐藏中,仅记录
    if (prev === 0 && h > 0) {                 // display:none 恢复
      const maxTop = maxTopOf(lane)
      if (lane.isFollowing) {
        programmaticScrollTo(lane, maxTop)     // 跟随:瞬移贴底,避免全程长滚
      } else {
        // 阅读态:Chrome/Safari 恢复显示时 scrollTop 被重置 0,用 lastWritten 恢复原位
        programmaticScrollTo(lane, Math.min(lane.lastWritten, maxTop))
      }
      updateJumpBtn(lane, maxTop - lane.el.scrollTop)
      return
    }
    updateJumpBtn(lane, maxTopOf(lane) - lane.el.scrollTop)
    if (lane.isFollowing) ensureLoop(lane)
  }

  function jumpToBottom(key: LaneKey) {
    const lane = lanes[key]
    if (lane) resumeFollow(lane)
  }

  onMounted(() => {
    const keys: LaneKey[] = ['chinese', 'english']
    for (const key of keys) {
      const sel = key === 'chinese' ? '.chinese-article.article-display' : '.english-article.article-display'
      const el = document.querySelector(sel) as HTMLElement | null
      if (!el || !el.isConnected) continue
      let lane!: LaneState
      const handleScroll = () => onScrollLane(lane)
      const handleWheel = (e: WheelEvent) => onWheelUp(lane, e)
      lane = {
        key,
        el,
        isFollowing: true,
        lastWritten: el.scrollTop,                              // 挂载初始化清单(spec 3.1)
        lastGeom: { h: el.scrollHeight, ch: el.clientHeight },
        userDrift: 0,
        running: false,
        raf: null,
        lastTs: 0,
        roLastH: el.scrollHeight,
        handleScroll,
        handleWheel
      }
      el.addEventListener('scroll', handleScroll, { passive: true })
      el.addEventListener('wheel', handleWheel, { passive: true })
      lanes[key] = lane
    }

    bodyRo = new ResizeObserver(() => {
      for (const key of keys) {
        const lane = lanes[key]
        if (lane) onBodyRoCheck(lane)
      }
    })
    boxRo = new ResizeObserver(() => {
      for (const key of keys) {
        const lane = lanes[key]
        // 容器盒变化(窗口 resize/fullscreen class):补无 scroll 事件场景的按钮刷新
        if (lane) updateJumpBtn(lane, maxTopOf(lane) - lane.el.scrollTop)
      }
    })
    for (const key of keys) {
      const lane = lanes[key]
      if (!lane) continue
      const body = lane.el.querySelector(':scope > .article-body')
      if (body) bodyRo.observe(body)
      boxRo.observe(lane.el)
    }

    mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    reducedMotion = mq.matches
    mqHandler = (e: MediaQueryListEvent) => { reducedMotion = e.matches }
    mq.addEventListener('change', mqHandler)

    for (const key of keys) {
      const lane = lanes[key]
      if (lane) {
        updateJumpBtn(lane, maxTopOf(lane) - lane.el.scrollTop)
        ensureLoop(lane)   // 初始跟随:首屏贴底
      }
    }
  })

  onBeforeUnmount(() => {
    for (const key of ['chinese', 'english'] as LaneKey[]) {
      const lane = lanes[key]
      if (!lane) continue
      cancelLoop(lane)
      lane.el.removeEventListener('scroll', lane.handleScroll)
      lane.el.removeEventListener('wheel', lane.handleWheel)
    }
    bodyRo?.disconnect()
    boxRo?.disconnect()
    if (syncRaf !== null) cancelAnimationFrame(syncRaf)
    if (mq && mqHandler) mq.removeEventListener('change', mqHandler)
  })

  return { showJumpCn, showJumpEn, jumpToBottom }
}
```

- [ ] **Step 2: 静态自检**

逐项核对（读回文件）：
1. 常量与 Global Constraints 完全一致（EPS=1/DRIFT=8/BOTTOM=1/JUMP_AT=52/TAU=300/MAX_DT=50）。
2. `stopFollow` 内含 `configSyncScroll` 分支联动停两栏。
3. `followLoop` 含 `lastTs` 初始化（ensureLoop）、逐帧更新、`MAX_DT` 钳制、reducedMotion 独立分支、贴底停帧清 `userDrift`。
4. `programmaticScrollTo` 是唯一程序写入点（全文无其他 `el.scrollTop =` 赋值）。
5. 卸载清理完整：cancelLoop ×N、双 RO disconnect、syncRaf、scroll/wheel listener、mq change。

Run: `cd /home/zhenyi/ownprojects/livetrans/frontend && yarn build`
Expected: 构建成功（esbuild 转译只暴露语法错误，不查类型——类型正确性靠 Step 2 逐项核对；此时尚无使用者，仅验证可编译）。

---

### Task 2: `ParagraphDisplayV2` 组件副本（.article-body 包裹层）

**Files:**
- Create: `frontend/components/content/ParagraphDisplayV2.vue`（复制 `ParagraphDisplay.vue` 后定点修改）

**Interfaces:**
- Consumes: props 与原版一致（`language`/`paragraphs`/`fontSize`）。
- Produces: 渲染结构 `.chinese-article.article-display > .article-body > .article-paragraph ×N`（Task 1 的 RO 观察目标）；**不再 emit `scroll`**。

- [ ] **Step 1: 复制文件**

```bash
cp /home/zhenyi/ownprojects/livetrans/frontend/components/content/ParagraphDisplay.vue \
   /home/zhenyi/ownprojects/livetrans/frontend/components/content/ParagraphDisplayV2.vue
```

- [ ] **Step 2: 定点修改（4 处）**

1. 模板第 2 行，删除 scroll 透传：

```html
<!-- 原 -->
<div :class="articleClasses" class="article-display" @scroll="$emit('scroll', $event)">
<!-- 改为 -->
<div :class="articleClasses" class="article-display">
```

2. v-for 块（第 3-45 行）整体包进内容包裹层——`<div ... class="article-display">` 开标签之后、其闭合 `</div>`（第 46 行）之前：

```html
<div :class="articleClasses" class="article-display">
  <div class="article-body">
    <div v-for="(paragraph, index) in processedParagraphs" ... >
      ...（原内容不动）...
    </div>
  </div>
</div>
```

3. `<script setup>` 中删除 `defineEmits(['scroll'])`（原第 68 行）。

4. `<style scoped>` 的 `.article-display` 规则块内追加两条声明（CSS 不变量）：

```css
.article-display {
  /* ...原有声明全部保留... */
  overflow-anchor: none;   /* 位置全由 useFollowScroll 管理,关闭浏览器锚定 */
}

/* 内容包裹层:RO 观察目标,flow-root 防子元素垂直 margin 折叠出观察盒 */
.article-body {
  display: flow-root;
  width: 100%;
}
```

- [ ] **Step 3: 静态自检**

读回确认：模板 `.article-body` 恰好包裹 v-for；`@scroll` 与 `defineEmits` 已不存在；`overflow-anchor` 已加；其余（props/computed/样式）与原版逐字一致。

---

### Task 3: `SectionHeaderV2` 组件副本（跳底按钮）

**Files:**
- Create: `frontend/components/content/SectionHeaderV2.vue`（复制 `SectionHeader.vue` 后定点修改）

**Interfaces:**
- Consumes: props `title`/`isFullscreen`/`fontSize`/`language`/`headerClasses`/`showJump: boolean`。
- Produces: emits `fullscreen`、`font-size-change`、`jump-to-bottom`。按钮仅在 `showJump` 为 true 时渲染。

- [ ] **Step 1: 复制文件**

```bash
cp /home/zhenyi/ownprojects/livetrans/frontend/components/content/SectionHeader.vue \
   /home/zhenyi/ownprojects/livetrans/frontend/components/content/SectionHeaderV2.vue
```

- [ ] **Step 2: 定点修改（5 处）**

1. 模板按钮区（原第 13-22 行）整体替换：

```html
<div class="auto-scroll-controls">
  <button
    v-show="showJump"
    class="font-size-btn auto-scroll-btn"
    @click="emit('jump-to-bottom')"
    title="回到最新 | Jump to latest"
    aria-label="回到最新 | Jump to latest"
  >
    <FontAwesomeIcon icon="arrow-down" />
  </button>
</div>
```

2. script 导入行，删除 `nextTick`：

```js
// 原
import { computed, nextTick } from 'vue'
// 改为
import { computed } from 'vue'
```

3. props：删除 `autoScroll` 定义块（原第 85-88 行），新增：

```js
  showJump: {
    type: Boolean,
    default: false
  }
```

4. emits 与点击处理：原第 91 行 `defineEmits` 替换为：

```js
const emit = defineEmits(['fullscreen', 'font-size-change', 'jump-to-bottom'])
```

原第 95-101 行（`handleAutoScrollClick` 函数）**整体删除**；原第 93 行 `fullscreenIcon` computed **原样保留**（勿重复声明）。

5. `<style scoped>` 末尾（`@media` 块之前）追加 focus-visible 样式（spec 2.8 无障碍要求）：

```css
.auto-scroll-btn:focus-visible {
  outline: 2px solid var(--primary-color, #00adb5);
  outline-offset: 2px;
}
```

- [ ] **Step 3: 样式与静态自检**

样式块其余部分**全部保留**（`.auto-scroll-btn` 系列作为按钮外观复用；无 `active` class 绑定后 pulse 动画自然不触发）。读回确认：
1. `autoScroll`、`handleAutoScrollClick`、`nextTick`、`toggle-auto-scroll`、`scroll-to-bottom` 字样均已不存在。
2. `v-show="showJump"`、`aria-label`、`:focus-visible` 样式存在。
3. `faArrowDown` 导入与 `library.add` 保留（按钮仍用 arrow-down 图标）。
4. `fullscreenIcon` computed 仅一处声明。

---

### Task 4: `ContentSectionV2` 组件副本（V2 接线）

**Files:**
- Create: `frontend/components/content/ContentSectionV2.vue`（复制 `ContentSection.vue` 后定点修改）

**Interfaces:**
- Consumes: 原 ContentSection 全部 props（`language`/`title`/`paragraphs`/`currentSegment`/`fontSize`/`sectionClasses`/`headerClasses`/`isFullscreen`/`isWaitingForService`/`lastCurrentEn`）**去掉 `autoScroll`**、新增 `showJump: boolean`；子组件 SectionHeaderV2/ParagraphDisplayV2（Task 2/3）与原版 CurrentInput。
- Produces: emits `fullscreen`、`font-size-change`、`jump-to-bottom`（payload 为空，语言由页面闭包区分）、`report`。

- [ ] **Step 1: 复制文件**

```bash
cp /home/zhenyi/ownprojects/livetrans/frontend/components/content/ContentSection.vue \
   /home/zhenyi/ownprojects/livetrans/frontend/components/content/ContentSectionV2.vue
```

- [ ] **Step 2: 定点修改（4 处）**

1. import 替换（原第 49-50 行）：

```js
import SectionHeaderV2 from './SectionHeaderV2.vue'
import ParagraphDisplayV2 from './ParagraphDisplayV2.vue'
```

（模板中 `<SectionHeader` → `<SectionHeaderV2`、`<ParagraphDisplay` → `<ParagraphDisplayV2` 两处标签同步替换；`CurrentInput` 不动。）

2. 模板 SectionHeader 调用处：`:auto-scroll="autoScroll"` → `:show-jump="showJump"`；`@toggle-auto-scroll="$emit('toggle-auto-scroll')"` 与 `@scroll-to-bottom="$emit('scroll-to-bottom')"` 两行删除，新增 `@jump-to-bottom="$emit('jump-to-bottom')"`。

3. 模板 ParagraphDisplayV2 调用处：删除 `@scroll="$emit('scroll', $event)"` 行。

4. props/emits：删除 `autoScroll` prop 块（原第 97-100 行），新增：

```js
  showJump: {
    type: Boolean,
    default: false
  }
```

`defineEmits` 改为：

```js
defineEmits(['fullscreen', 'font-size-change', 'jump-to-bottom', 'report'])
```

- [ ] **Step 3: 静态自检**

读回确认：`autoScroll`、`@scroll`、`toggle-auto-scroll`、`scroll-to-bottom` 均不存在；`showJump`/`jump-to-bottom` 接线完整；welcome/current/report 部分与原版一致。

---

### Task 5: `pages/st.vue` 测试页

**Files:**
- Create: `frontend/pages/st.vue`（路由 `/st`，不进任何导航）

**Interfaces:**
- Consumes: `useFollowScroll`（Task 1）、`ContentSectionV2`（Task 4）、现有 `useAppConfig`/`useWebSocket`/`useParagraphLogic`/`useFullscreen`/`LayoutAppHeader`/`ContentSectionDivider`/`CommonReportDialog`。
- Produces: 无（叶子页面）。

- [ ] **Step 1: 复制原页**

```bash
cp /home/zhenyi/ownprojects/livetrans/frontend/pages/index.vue \
   /home/zhenyi/ownprojects/livetrans/frontend/pages/st.vue
```

- [ ] **Step 2: 定点修改（script 部分）**

1. `useAppConfig()` 解构中删除 `configAutoScroll, toggleAutoScroll,` 两行（其余保留）。
2. 新增占位 ref（AppHeader 的 `configAutoScroll` prop 为 required，测试页菜单开关不生效，仅占位）：

```js
// /st 对比测试页:自动滚动开关占位(新机制无开关),菜单里点击无效属预期
const placeholderAutoScroll = ref(true)
```

3. `useWebSocket(...)` 调用去掉初始化回调参数（原第 28-33 行整个回调函数删除）：

```js
const {
  wsConnected,
  currentSegment,
  lastCurrentEn,
  confirmedSegments,
  isWaitingForService
} = useWebSocket()
```

4. `useScrollSync(...)` 解构整块（原第 40-45 行）删除，替换为：

```js
const { showJumpCn, showJumpEn, jumpToBottom } = useFollowScroll(configSyncScroll)
```

5. 删除 `watchDataAndScroll(currentSegment, confirmedSegments)` 调用行（原第 60 行）及其上一行陈旧注释 `// 监听数据变化并自动滚动`（原第 59 行）。

- [ ] **Step 3: 定点修改（template 部分）**

1. `LayoutAppHeader`：`:config-auto-scroll="configAutoScroll"` → `:config-auto-scroll="placeholderAutoScroll"`；删除 `@toggle-auto-scroll="toggleAutoScroll"` 与 `@scroll-to-bottom="scrollToBottom"` 两行；其余绑定不动。
2. 中文栏 `<ContentSection` → `<ContentSectionV2`；删除 `:auto-scroll="configAutoScroll"`、`@scroll="onChineseScroll"`、`@toggle-auto-scroll="toggleAutoScroll"`、`@scroll-to-bottom="scrollToBottom"` 四行；新增 `:show-jump="showJumpCn"` 与 `@jump-to-bottom="jumpToBottom('chinese')"`。
3. 英文栏同上改造：标签换 `ContentSectionV2`；删除 `:auto-scroll`/`@scroll="onEnglishScroll"`/`@toggle-auto-scroll`/`@scroll-to-bottom`；新增 `:show-jump="showJumpEn"` 与 `@jump-to-bottom="jumpToBottom('english')"`。
4. `<ContentSectionDivider>`、`<CommonReportDialog>`、两个 `<style>` 块不动。

- [ ] **Step 4: 静态自检**

读回确认：`configAutoScroll`/`toggleAutoScroll`/`scrollToBottom`/`onChineseScroll`/`onEnglishScroll`/`watchDataAndScroll`/`useScrollSync`/`<ContentSection `（无 V2 后缀的）在本文件中零残留；`placeholderAutoScroll`/`showJumpCn`/`showJumpEn`/`jumpToBottom`/`ContentSectionV2` 全部就位；`<style>` 两块与原页一致。

---

### Task 6: 构建验证与手动验收清单

**Files:**
- 无新文件（验证任务）

- [ ] **Step 1: 构建**

Run: `cd /home/zhenyi/ownprojects/livetrans/frontend && yarn build`
Expected: 构建成功无错误。若失败：修复后重跑直至通过（常见问题：组件 import 路径大小写、模板未闭合标签）。

- [ ] **Step 2: 路由可达性冒烟（dev 模式可选）**

Run: `cd /home/zhenyi/ownprojects/livetrans/frontend && yarn dev` 后浏览器开 `http://localhost:3000/st`
Expected: 页面渲染与 `/` 视觉一致（含暗色主题）；浏览器 console 无 `useFollowScroll`/`ContentSectionV2` 相关报错；`http://localhost:3000/` 行为与改动前完全一致（零改动验证）。验证完停掉 dev。

- [ ] **Step 3: 输出手动验收清单与偏差明细（交付物注释，不阻塞）**

在完成报告中附两份内容：
1. spec 第五节 20 条验收清单，标注"待用户在真实直播流/dev mock 下执行"——本计划不含自动化验收（无测试基建），手动验收是用户审核的一部分。
2. **副本策略 vs spec 三节的偏差明细**（供用户审核对照）：保留项——`useScrollSync.ts` 未删、原四组件未改、AppHeader"自动滚动"菜单项仍在（测试页点击无效）、`configAutoScroll` 在 `/` 页仍可写 localStorage、`faArrowDown`/`.auto-scroll-btn` 样式在 V2 中保留复用；新增项——`useFollowScroll.ts`、三个 `*V2.vue`、`pages/st.vue`。

---

## Self-Review 记录

1. **Spec 覆盖**：spec 二节机制（2.1 状态/LaneState→Task1、2.2 协议→Task1、2.3/2.4 停/恢复→Task1、2.5 动画→Task1、2.6 RO/按钮→Task1+2、2.7 联动→Task1、2.8 按钮→Task3、2.9 监听裁剪→Task1）全部有 task；三节改动清单以"副本"形式覆盖（原文件不动的差异见各 Task 定点修改，`useScrollSync.ts`/`useAppConfig.ts`/原组件按副本策略**不删不改**，属预期偏差并在完成报告标注）；3.2 CSS 不变量→Task2；四节边界→Task1 代码注释对应；五节验收→Task6。无缺口。
2. **占位符扫描**：无 TBD/TODO/"适当处理"；副本任务以 cp + 精确 diff 表达，可无歧义执行。
3. **类型/命名一致性**：`showJumpCn`/`showJumpEn`/`jumpToBottom(key)`（Task1 产出 = Task5 消费）、`showJump` prop（Task3/4 同名）、`jump-to-bottom` event（Task3/4/5 同名）、`.article-body`（Task2 产出 = Task1 `:scope > .article-body` 查询）一致。
