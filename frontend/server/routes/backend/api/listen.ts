// server/api/listen.ts
// import { defineEventHandler, readBody } from 'h3'
import {
  Segment,
  saveCurrentSegment,
  isValidSegment,
} from '../../../utils/segments'

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

        await seg.processText(contextSegs)
        broadcast({
          current_en: seg,
        })
      }
    }
  }

  if (data.confirmed) {
    // console.log('Confirmed data:', data.confirmed)
    let segs = data.confirmed
      .map((seg: Segment) => {
        seg.text = cnT2S(seg.text)
        return Segment.fromObject(seg)
      })
      .filter((seg: Segment) => isValidSegment(seg))
    if (segs.length > 0) {
      // console.log('---Confirmed segments:', segs)
      segs.forEach(async (seg: Segment) => {
        let contextSegs = saveConfirmedSegment(seg)
        broadcast({
          confirmed: seg,
        })

        await seg.processText(contextSegs);
        broadcast({
          update: seg,
        })
      })
    }
  }

  return { message: 'Data received successfully' }
})
