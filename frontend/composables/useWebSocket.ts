import { ref, reactive, computed, onMounted, onUnmounted, nextTick } from 'vue'

export function useWebSocket(onInitCallback = null) {
  const ws = ref(null)
  const wsConnected = ref(false)
  const currentSegment = ref({ text: "" })
  const lastCurrentEn = ref("")
  const confirmedSegments = reactive({ value: [] })

  const isWaitingForService = computed(() => {
    return (
      currentSegment.value.text === "" &&
      confirmedSegments.value.length === 0
    )
  })

  const reconnectWS = () => {
    console.log("Reconnecting WebSocket...")
    setTimeout(connectWS, 1000)
  }

  const connectWS = () => {
    ws.value = new WebSocket("/backend/api/ws")
    console.log("Establishing WebSocket connection...")

    const timeoutId = setTimeout(() => {
      if (ws.value.readyState === WebSocket.CONNECTING) {
        console.log("WebSocket connection timed out, retrying in 3 seconds...")
        ws.value.close()
      }
    }, 5000)

    // 连接成功处理
    ws.value.onopen = () => {
      console.log("WebSocket connection established")
      wsConnected.value = true
    }

    // 连接错误处理
    ws.value.onerror = (error) => {
      console.log("WebSocket error observed:", error)
      wsConnected.value = false
    }

    // 连接关闭处理
    ws.value.onclose = (event) => {
      console.log("WebSocket connection closed", event)
      wsConnected.value = false
      reconnectWS()
    }

    ws.value.onmessage = (event) => {
      let data = JSON.parse(event.data)

      if (data.init) {
        currentSegment.value = data.init.current
        confirmedSegments.value = data.init.confirmed

        // 触发初始化回调
        if (onInitCallback && typeof onInitCallback === 'function') {
          nextTick(() => {
            onInitCallback()
          })
        }
      }
      if (data.current) {
        currentSegment.value = data.current
      }
      if (data.current_en) {
        lastCurrentEn.value = data.current_en.en_text
      }
      if (data.confirmed) {
        confirmedSegments.value.push(data.confirmed)
      }
      if (data.update) {
        const index = confirmedSegments.value.findLastIndex((s) => {
          return s.id === data.update.id
        })

        if (index >= 0) {
          confirmedSegments.value[index] = data.update
        }
      }
    }
  }

  onMounted(() => {
    connectWS()
  })

  return {
    ws,
    wsConnected,
    currentSegment,
    lastCurrentEn,
    confirmedSegments,
    isWaitingForService,
    connectWS,
    reconnectWS
  }
}