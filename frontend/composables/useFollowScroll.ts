// composables/useFollowScroll.ts
// 自动跟随滚动 v2 —— spec: docs/2026-09-16_自动跟随滚动重构spec_取消手动开关.md
// 核心: scroll 事件三分类(布局指纹/程序写入/外部位移) + 追赶式缓动 + 联动 rAF 合并器
import { ref, onMounted, onBeforeUnmount, type Ref } from 'vue'

// ===== spec 阈值常量 =====
const EPS = 1        // 程序写入自身事件的噪声容差(px)
const DRIFT = 8      // 向上累计介入阈值(px)
const BOTTOM = 1     // 贴底判定(px)
const JUMP_AT = 52   // 跳底按钮显隐阈值(px)
const TAU = 600      // 追赶动画时间常数(ms):越大缓动越慢越可感知;日常单条字幕更新(30~60px)
                     // 约0.7s走完,有明确滚动感;持续流稳态滞后≈增长速度v·τ(约60px)
const MAX_DT = 50    // 单帧 dt 上限(ms),防后台标签页回前台 k≈1 瞬移

type LaneKey = 'chinese' | 'english'

interface LaneState {
  key: LaneKey
  el: HTMLElement
  isFollowing: boolean
  lastWritten: number                    // 最后已知 scrollTop(程序写入读回值/外部位移新基线)
  savedTop: number                       // display:none 隐藏瞬间的位置存档(恢复原位用,防 lastWritten 被迟到事件污染)
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
  let disposed = false
  // 注:不做 prefers-reduced-motion 瞬移分支——实测开发者/用户环境(Edge 关闭动画设置)
  // 会命中系统 reduce,导致"永远瞬移无动画"且排查困难;直播展示页场景放弃该无障碍特性

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

  function resumeFollowLocal(lane: LaneState) {
    lane.isFollowing = true
    lane.userDrift = 0
    lane.lastWritten = Math.min(Math.max(lane.el.scrollTop, 0), maxTopOf(lane))
    ensureLoop(lane)
    updateJumpBtn(lane, maxTopOf(lane) - lane.lastWritten)
  }

  // 恢复与停止对称:联动开启时一栏贴底自愈,另一栏一并恢复并追赶贴底——
  // 否则被联动拖离底部的栏没有任何恢复路径(spec v2.4 的单栏恢复仅在联动关闭时成立)
  function resumeFollow(source: LaneState) {
    resumeFollowLocal(source)
    if (configSyncScroll.value) {
      const peer = other(source)
      if (peer && !peer.isFollowing) resumeFollowLocal(peer)
    }
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
    programmaticScrollTo(lane, el.scrollTop + dist * (1 - Math.exp(-dt / TAU)))
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
    const el = lane.el
    const h = el.scrollHeight
    const prev = lane.roLastH
    lane.roLastH = h
    if (prev > 0 && h === 0) {                 // 隐藏瞬间:存档位置,恢复时以 savedTop 为准
      lane.savedTop = lane.lastWritten
      return
    }
    if (prev === 0 && h === 0) return          // 隐藏中,仅记录
    if (prev === 0 && h > 0) {                 // display:none 恢复
      const maxTop = maxTopOf(lane)
      if (lane.isFollowing) {
        programmaticScrollTo(lane, maxTop)     // 跟随:瞬移贴底,避免全程长滚
      } else {
        // 阅读态:Chrome/Safari 恢复显示时 scrollTop 被重置 0,用 savedTop 恢复原位
        programmaticScrollTo(lane, Math.min(lane.savedTop, maxTop))
      }
      // 同步指纹与漂移:布局诱发的迟到 scroll 事件自此走②分支,不误判介入/不丢原位
      lane.lastGeom = { h: el.scrollHeight, ch: el.clientHeight }
      lane.userDrift = 0
      updateJumpBtn(lane, maxTop - el.scrollTop)
      return
    }
    updateJumpBtn(lane, maxTopOf(lane) - el.scrollTop)
    if (lane.isFollowing) ensureLoop(lane)
  }

  function jumpToBottom(key: LaneKey) {
    const lane = lanes[key]
    if (lane) resumeFollow(lane)
  }

  // 页面由 <ClientOnly> 包裹:插槽内容在父组件 onMounted 之后的重渲染周期才插入 DOM,
  // 首次 querySelector 必为 null——轮询重试直到两栏容器就位再挂监听/RO(修复 /st 空转)
  onMounted(() => { initLanes(0) })

  function initLanes(attempt: number) {
    if (disposed) return
    const keys: LaneKey[] = ['chinese', 'english']
    for (const key of keys) {
      if (lanes[key]) continue
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
        savedTop: el.scrollTop,
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

    const missing = keys.some(key => !lanes[key])
    if (missing) {
      if (attempt >= 20) return   // ~2s 仍未就位(异常布局),放弃避免无限轮询
      setTimeout(() => initLanes(attempt + 1), 100)
      return
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

    for (const key of keys) {
      const lane = lanes[key]
      if (lane) {
        updateJumpBtn(lane, maxTopOf(lane) - lane.el.scrollTop)
        ensureLoop(lane)   // 初始跟随:首屏贴底
      }
    }
  }

  onBeforeUnmount(() => {
    disposed = true
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
  })

  return { showJumpCn, showJumpEn, jumpToBottom }
}
