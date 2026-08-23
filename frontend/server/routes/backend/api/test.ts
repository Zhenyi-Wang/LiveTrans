// import { broadcast, getPeers } from "../utils/ws";

import { aiProcessText } from "../../../utils/ai";

// export default defineEventHandler(()=>{
//   broadcast({current: "current", confirmed: "confirmed"}  )
//   return 1;
// })

export default defineEventHandler(async (event) => {
  const body = await readBody(event);
  const texts: string[] = body.texts || [body.text || "圣经真正的名字是旧兴曰全书。"];
  const context = body.context || [];

  const results = await aiProcessText(context, texts);
  return results.map((r, i) => ({
    original: texts[i],
    optimized: r.optimized,
    translated: r.translated,
  }));
});
