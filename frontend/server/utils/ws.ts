import {Peer} from "crossws"

let peers: Peer[] = []

export function savePeer(peer: Peer) {
  peers.push(peer)
}

export function getPeers(): Peer[] {
  return peers
}

export function removePeer(peer: Peer) {
  peers = peers.filter(p => p.id !== peer.id)
}

export function broadcast(message: any) {
  for (let peer of peers) {
    peer.send(message)
  }
}