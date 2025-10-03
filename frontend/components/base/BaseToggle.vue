<template>
  <label class="base-toggle">
    <input
      type="checkbox"
      :checked="modelValue"
      @change="$emit('update:modelValue', $event.target.checked)"
      :disabled="disabled"
      class="base-toggle__input"
    />
    <span class="base-toggle__slider"></span>
  </label>
</template>

<script setup>
defineProps({
  modelValue: {
    type: Boolean,
    default: false
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

defineEmits(['update:modelValue'])
</script>

<style scoped>
.base-toggle {
  position: relative;
  display: inline-block;
  width: 52px;
  height: 28px;
  cursor: pointer;
}

.base-toggle__input {
  opacity: 0;
  width: 0;
  height: 0;
  position: absolute;
}

.base-toggle__slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #cbd5e1;
  transition: all 0.3s ease;
  border-radius: 28px;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
}

.base-toggle__slider:before {
  position: absolute;
  content: "";
  height: 20px;
  width: 20px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  transition: all 0.3s ease;
  border-radius: 50%;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}

.base-toggle__input:checked + .base-toggle__slider {
  background-color: var(--primary-color, #00adb5);
  box-shadow: 0 0 16px rgba(0, 173, 181, 0.4);
}

.base-toggle__input:checked + .base-toggle__slider:before {
  transform: translateX(24px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.base-toggle__slider:hover {
  background-color: #94a3b8;
}

.base-toggle__input:checked + .base-toggle__slider:hover {
  background-color: var(--primary-hover, #00c4cc);
}

.base-toggle:has(.base-toggle__input:disabled) {
  cursor: not-allowed;
  opacity: 0.5;
}

.base-toggle:has(.base-toggle__input:disabled) .base-toggle__slider {
  cursor: not-allowed;
}
</style>