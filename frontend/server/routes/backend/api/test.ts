// import { broadcast, getPeers } from "../utils/ws";

import { aiOptiText } from "../../../utils/ai";

// export default defineEventHandler(()=>{
//   broadcast({current: "current", confirmed: "confirmed"}  )
//   return 1;
// })

export default defineEventHandler(async (event) => {
  return {
    text: await aiOptiText(
      "严格地说,圣经只不过是一个俗称,通俗的讲法。 圣经真正的名字是什么?",
      "圣经真正的名字是旧兴曰全书。"
    ),
  };
  // let text = getQuery(event).text as string;

  // return {text:cnT2S(text)};
});
