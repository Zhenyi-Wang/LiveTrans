import { getConfirmedSegments } from "../../../utils/segments";
import { removePeer, savePeer } from "../../../utils/ws";

export let peerPub: any = null;

export default defineWebSocketHandler({
  open(peer) {
    console.log("[ws] open", peer);
    savePeer(peer)
    let initMsg = {
      init:{
        confirmed: getConfirmedSegments(),
        current: getCurrentSegment()
      }
    }
    peer.send(initMsg)
  },

  message(peer, message) {
    console.log("[ws] message", peer, message);
  },

  close(peer, event) {
    removePeer(peer)
    console.log("[ws] close", peer, event);
  },

  error(peer, error) {
    console.log("[ws] error", peer, error);
  },
});
