// import { broadcast, getPeers } from "../utils/ws";

import { clearAllSegments } from "~/server/utils/segments";

// export default defineEventHandler(()=>{
//   broadcast({current: "current", confirmed: "confirmed"}  )
//   return 1;
// })

export default defineEventHandler(async (event) => {
  clearAllSegments()
  return {
  
  };
  // let text = getQuery(event).text as string;

  // return {text:cnT2S(text)};
});
