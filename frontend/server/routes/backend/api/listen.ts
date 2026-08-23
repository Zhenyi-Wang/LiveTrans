// server/api/listen.ts
// import { defineEventHandler, readBody } from 'h3'
import {
  Segment,
  saveCurrentSegment,
  getCurrentSegment,
  isValidSegment,
  enqueueConfirmed,
} from '../../../utils/segments'

// ===== current预览控制:最新者胜,无排队 =====
// preview并发=1(与confirmed drain串行的1相加,LLM总并发≤2);
// in-flight期间新current只记入pending槽(覆盖旧值=忽略中间版本),
// 上一个完成后翻译pending里最新的那个——天然背压,永不积压,永远翻最新
let previewActive = false
let pendingSeg: Segment | null = null

// 预览缓存:转录抖动导致current文本来回变化(A→B→A)时直接复用,避免重复LLM调用
const previewCache = new Map<string, string>()
const PREVIEW_CACHE_MAX = 50

function broadcastIfFresh(seg: Segment): void {
  if (seg.en_text && seg.en_text !== seg.text
    && String(getCurrentSegment().start) === String(seg.start)) {
    broadcast({
      current_en: seg,
    })
  }
}

async function previewCurrent(seg: Segment): Promise<void> {
  if (previewActive) {
    pendingSeg = seg
    return
  }
  previewActive = true
  try {
    while (seg) {
      const cached = previewCache.get(seg.text)
      if (cached !== undefined) {
        seg.en_text = cached
      } else {
        await seg.previewInput()
        if (seg.en_text && seg.en_text !== seg.text) {
          if (previewCache.size >= PREVIEW_CACHE_MAX) {
            const oldest = previewCache.keys().next()
            if (!oldest.done) previewCache.delete(oldest.value)
          }
          previewCache.set(seg.text, seg.en_text)
        }
      }
      broadcastIfFresh(seg)
      seg = pendingSeg
      pendingSeg = null
    }
  } finally {
    previewActive = false
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
        saveCurrentSegment(data.current)

        // 异步预翻不阻塞响应(dispatch消费速度取决于POST响应时间);
        // 最新者胜:处理中只保留最新pending,完成后续翻最新版
        void previewCurrent(seg)
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
