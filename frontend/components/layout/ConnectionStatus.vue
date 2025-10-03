<template>
  <div class="main-connection-status">
    <div :class="connectionDotClasses"></div>
  </div>
</template>

<script setup>
const props = defineProps({
  connected: {
    type: Boolean,
    required: true
  }
})

const connectionDotClasses = computed(() => [
  'connection-dot',
  props.connected ? 'connected' : 'disconnected'
])
</script>

<style scoped>
.main-connection-status {
  display: flex;
  align-items: center;
  margin-right: 15px;
}

.connection-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: #00ff55;
  box-shadow: 0 0 8px rgba(0, 255, 85, 0.4);
  transition: all 0.3s ease;
}

.connection-dot.connected {
  background-color: #00ff55;
  box-shadow: 0 0 8px rgba(0, 255, 85, 0.4);
}

.connection-dot.disconnected {
  background-color: #ff4444;
  box-shadow: 0 0 8px rgba(255, 68, 68, 0.4);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.7;
    transform: scale(1.1);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
}
</style>