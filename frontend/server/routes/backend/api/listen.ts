// server/api/listen.ts
// import { defineEventHandler, readBody } from 'h3'
import {
  Segment,
  saveCurrentSegment,
  getCurrentSegment,
  isValidSegment,
  enqueueConfirmed,
} from '../../../utils/segments'

// ===== current预览控制 =====
// preview并发=1(与confirmed drain串行的1相加,LLM总并发≤2,低于网关3并发上限)
let previewActive = false
const previewWaiters: (() => void)[] = []

// 预览缓存:转录抖动导致current文本来回变化(A→B→A)时直接复用,避免重复LLM调用
const previewCache = new Map<string, string>()
const PREVIEW_CACHE_MAX = 50

async function previewCurrent(seg: Segment, contextSegs: Segment[]) {
  const cached = previewCache.get(seg.text)
  if (cached !== undefined) {
    seg.en_text = cached
    return
  }
  // 队列已积压:本条必然过时,直接放弃(LLM稍慢于current变化频率时防无界积压)
  if (previewWaiters.length >= 2) return
  if (previewActive) {
    await new Promise<void>(resolve => previewWaiters.push(resolve))
    // 排到执行:current已切句则本条过时,放弃
    if (String(getCurrentSegment().start) !== String(seg.start)) return
  }
  previewActive = true
  try {
    await seg.previewInput(contextSegs)
    if (seg.en_text && seg.en_text !== seg.text) {
      if (previewCache.size >= PREVIEW_CACHE_MAX) {
        const oldest = previewCache.keys().next()
        if (!oldest.done) previewCache.delete(oldest.value)
      }
      previewCache.set(seg.text, seg.en_text)
    }
  } finally {
    previewActive = false
    const next = previewWaiters.shift()
    if (next) next()
  }
}

export default defineEventHandler(async event => {
  const data = await readBody(event)
  console.log('Received data:', typeof data, data)

  if (data.current) {
    console
    let needBroadcast = getCurrentSegment().text !== data.current.text
    if (needBroadcast) {
      data.current.text = cnT2S(data.current.text)
      let seg = Segment.fromObject(data.current)
      if (isValidSegment(seg)) {
        broadcast({
          current: data.current,
        })
        let contextSegs = saveCurrentSegment(data.current)

        // 异步预翻不阻塞响应(dispatch消费速度取决于POST响应时间);
        // 排队超时放弃;超时/失败未产出翻译(en_text空)不广播;
        // 新鲜度按句子start判断:同句滚动演进可广播,跨句才拦
        void Promise.race([
          previewCurrent(seg, contextSegs),
          new Promise(r => setTimeout(r, 12000)),
        ]).then(() => {
          if (seg.en_text && String(getCurrentSegment().start) === String(seg.start)) {
            broadcast({
              current_en: seg,
            })
          }
        })
      }
    }
  }

  if (data.confirmed) {
    let segs = data.confirmed
      .map((seg: Segment) => {
        seg.text = cnT2S(seg.text)
        return Segment.fromObject(seg)
      })
      .filter((seg: Segment) => isValidSegment(seg))
    if (segs.length > 0) {
      segs.forEach((seg: Segment) => {
        enqueueConfirmed(seg)
        broadcast({
          confirmed: seg,
        })
      })
    }
  }

  return { message: 'Data received successfully' }
})
