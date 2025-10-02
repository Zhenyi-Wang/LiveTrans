// import { broadcast, getPeers } from "../utils/ws";

import { aiProcessText } from "../../../utils/ai";

// export default defineEventHandler(()=>{
//   broadcast({current: "current", confirmed: "confirmed"}  )
//   return 1;
// })

export default defineEventHandler(async (event) => {
  const body = await readBody(event);
  const text = body.text || "圣经真正的名字是旧兴曰全书。";
  const context = body.context || "严格地说,圣经只不过是一个俗称,通俗的讲法。 圣经真正的名字是什么?";

  const result = await aiProcessText(context, text);
  return {
    original: text,
    optimized: result.optimized,
    translated: result.translated,
  };
});
