<template>
  <button
    :class="buttonClasses"
    @click="$emit('click', $event)"
    :disabled="disabled"
  >
    <slot />
  </button>
</template>

<script setup>
const props = defineProps({
  variant: {
    type: String,
    default: 'default',
    validator: (value) => ['default', 'primary', 'secondary', 'icon'].includes(value)
  },
  size: {
    type: String,
    default: 'medium',
    validator: (value) => ['small', 'medium', 'large'].includes(value)
  },
  disabled: {
    type: Boolean,
    default: false
  },
  active: {
    type: Boolean,
    default: false
  }
})

defineEmits(['click'])

const buttonClasses = computed(() => [
  'base-button',
  `base-button--${props.variant}`,
  `base-button--${props.size}`,
  {
    'base-button--disabled': props.disabled,
    'base-button--active': props.active
  }
])
</script>

<style scoped>
.base-button {
  background: none;
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: inherit;
}

.base-button:focus {
  outline: 2px solid var(--primary-color, #00adb5);
  outline-offset: 2px;
}

/* 尺寸变体 */
.base-button--small {
  padding: 4px 8px;
  font-size: 0.875rem;
  min-width: 32px;
  min-height: 24px;
}

.base-button--medium {
  padding: 8px 16px;
  font-size: 1rem;
  min-width: 44px;
  min-height: 32px;
}

.base-button--large {
  padding: 12px 24px;
  font-size: 1.125rem;
  min-width: 56px;
  min-height: 44px;
}

.base-button--icon {
  padding: 8px;
  border-radius: 8px;
}

/* 风格变体 */
.base-button--default {
  background-color: var(--bg-color, #ffffff);
  color: var(--text-color, #2c3e50);
}

.base-button--default:hover {
  background-color: var(--hover-bg, #f7fafc);
}

.base-button--primary {
  background: linear-gradient(135deg, var(--primary-color, #00adb5) 0%, var(--primary-hover, #00c4cc) 100%);
  color: white;
  border-color: var(--primary-color, #00adb5);
  box-shadow: 0 2px 8px rgba(0, 173, 181, 0.3);
}

.base-button--primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 173, 181, 0.4);
}

.base-button--secondary {
  background: transparent;
  border: 2px solid var(--border-color, #e2e8f0);
  color: var(--text-color, #2c3e50);
}

.base-button--secondary:hover {
  background-color: var(--hover-bg, #f7fafc);
  border-color: var(--primary-color, #00adb5);
}

/* 状态 */
.base-button--disabled {
  opacity: 0.5;
  cursor: not-allowed;
  pointer-events: none;
}

.base-button--active {
  background-color: var(--primary-color, #00adb5);
  color: white;
  border-color: var(--primary-color, #00adb5);
}

/* 动画 */
.base-button:not(.base-button--disabled):active {
  transform: translateY(0);
}

/* 暗色主题 */
.dark .base-button--default {
  background-color: var(--bg-color, #1a202c);
  border-color: var(--border-color, #4a5568);
}

.dark .base-button--default:hover {
  background-color: var(--hover-bg, #2d3748);
}

.dark .base-button--secondary {
  border-color: var(--border-color, #4a5568);
  color: var(--text-color, #e2e8f0);
}

.dark .base-button--secondary:hover {
  background-color: var(--hover-bg, #2d3748);
  border-color: var(--primary-color, #00adb5);
}
</style>