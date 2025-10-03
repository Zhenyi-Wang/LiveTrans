<template>
  <div class="connection-status">
    <div :class="connectionClass"></div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  connected: {
    type: Boolean,
    required: true
  }
})

const connectionClass = computed(() => ({
  'connection-dot': true,
  'connected': props.connected,
  'disconnected': !props.connected
}))
</script>

<style scoped>
.connection-status {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
}

.connection-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #ccc;
  transition: all 0.3s ease;
  box-shadow: 0 0 0 1px rgba(0, 173, 181, 0.1);
}

.connection-dot.connected {
  background-color: #00adb5;
  animation: pulse 2s infinite;
}

.connection-dot.disconnected {
  background-color: #f38181;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(0, 173, 181, 0.4);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(0, 173, 181, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(0, 173, 181, 0);
  }
}
</style>