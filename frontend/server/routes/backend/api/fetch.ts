// server/api/listen.ts
// import { defineEventHandler, readBody } from 'h3'

export default defineEventHandler(async (event) => {

  let data = await useStorage().getItem<any[]>('receivedData') || []
  return data
})