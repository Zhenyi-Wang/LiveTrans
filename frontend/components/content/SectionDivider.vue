<template>
  <div class="divider" :class="dividerClasses"></div>
</template>

<script setup>
defineProps({
  dividerClasses: {
    type: Object,
    required: true
  }
})
</script>

<style scoped>
.divider {
  width: 100%;
  height: 2px;
  background: linear-gradient(90deg,
    transparent 0%,
    var(--border-color, #e2e8f0) 20%,
    var(--border-color, #e2e8f0) 80%,
    transparent 100%
  );
  flex-shrink: 0;
  transition: all 0.3s ease;
}

.divider.hidden {
  opacity: 0;
  height: 0;
  margin: 0;
}

/* 深色主题 */
.dark .divider {
  background: linear-gradient(90deg,
    transparent 0%,
    #4a5568 20%,
    #4a5568 80%,
    transparent 100%
  );
}
</style>