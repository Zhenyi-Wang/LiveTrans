<template>
  <div v-show="visible" class="report-overlay" @click="$emit('close')">
    <div class="report-panel" role="dialog" aria-modal="true" aria-label="Report an Issue" @click.stop>
      <div class="report-header">
        <h3>
          <span class="report-title-icon"><FontAwesomeIcon icon="comment-dots" /></span>
          Report an Issue
        </h3>
        <button class="report-close-btn" aria-label="Close" @click="$emit('close')">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>

      <div class="report-body">
        <template v-if="status !== 'sent'">
          <p class="report-hint-text">
            Something not working? Tell us what happened.
            <br />
            <span class="report-hint-sub">(optional — you can send it empty)</span>
          </p>
          <textarea
            ref="textareaEl"
            v-model="description"
            class="report-textarea"
            :disabled="status === 'sending'"
            maxlength="500"
            rows="4"
            placeholder="Describe the problem..."
          ></textarea>
          <div class="report-char-count">{{ description.length }}/500</div>
        </template>

        <div v-else class="report-success">
          <span class="report-success-icon">✓</span>
          <p>Thanks! Your report has been sent.</p>
        </div>

        <div v-if="status === 'error'" class="report-error">
          {{ errorText }}
        </div>

        <button class="report-send-btn" :disabled="sendDisabled" @click="onSendClick">
          <template v-if="status === 'sending'">Sending...</template>
          <template v-else-if="status === 'sent'">Send Another</template>
          <template v-else>Send</template>
        </button>
        <div v-if="countdown > 0" class="report-countdown-hint">
          Please wait {{ countdown }}s before sending again.
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onUnmounted, computed, nextTick } from 'vue'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { faCommentDots } from '@fortawesome/free-solid-svg-icons'
import { library } from '@fortawesome/fontawesome-svg-core'

library.add(faCommentDots)

// 与后端 report.ts 的 RATE_LIMIT_MS 保持一致
const RATE_LIMIT_SECONDS = 30
const STORAGE_KEY = 'reportLastSentAt'

const props = defineProps({
  visible: {
    type: Boolean,
    required: true
  }
})

const emit = defineEmits(['close'])

const description = ref('')
const status = ref('idle') // idle | sending | sent | error
const errorText = ref('')
const countdown = ref(0)
const textareaEl = ref(null)
let timer = null
let requestSeq = 0 // 守卫: 弹窗关闭重开后丢弃在途请求的结果

const sendDisabled = computed(() => status.value === 'sending' || countdown.value > 0)

const stopTimer = () => {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

const startCountdown = (seconds) => {
  countdown.value = seconds
  stopTimer()
  timer = setInterval(() => {
    countdown.value -= 1
    if (countdown.value <= 0) {
      countdown.value = 0
      stopTimer()
    }
  }, 1000)
}

// 剩余锁定秒数(localStorage 持久, 刷新后锁定仍在); 读不到存储按未锁定处理
const remainingLockSeconds = () => {
  try {
    const last = Number(localStorage.getItem(STORAGE_KEY) || 0)
    const elapsed = (Date.now() - last) / 1000
    return elapsed < RATE_LIMIT_SECONDS ? Math.ceil(RATE_LIMIT_SECONDS - elapsed) : 0
  } catch {
    return 0
  }
}

const markSentLocally = () => {
  try { localStorage.setItem(STORAGE_KEY, String(Date.now())) } catch {}
}

const onEscKey = (e) => {
  if (e.key === 'Escape') emit('close')
}

watch(
  () => props.visible,
  (opened) => {
    if (opened) {
      status.value = 'idle'
      errorText.value = ''
      description.value = ''
      const remaining = remainingLockSeconds()
      if (remaining > 0) startCountdown(remaining)
      window.addEventListener('keydown', onEscKey)
      nextTick(() => textareaEl.value?.focus())
    } else {
      // 关闭即使途请求的 seq 过期, 防重开后旧结果落地清空新输入/误报状态
      requestSeq++
      window.removeEventListener('keydown', onEscKey)
    }
  }
)

onUnmounted(() => {
  stopTimer()
  window.removeEventListener('keydown', onEscKey)
})

// sent 态下 textarea 已隐藏且描述已清空, 按钮只负责回到输入态, 不再提交(防空报告)
const onSendClick = () => {
  if (sendDisabled.value) return
  if (status.value === 'sent') {
    status.value = 'idle'
    return
  }
  submit()
}

const submit = async () => {
  status.value = 'sending'
  const seq = ++requestSeq
  try {
    const resp = await fetch('/backend/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: description.value })
    })
    if (seq !== requestSeq || !props.visible) return // 重开后旧结果作废

    if (resp.status === 429) {
      // 30s 内已有同网络提交: 中性提示 + 按剩余锁定(无本地记录时保守锁满窗口防连点)
      errorText.value = 'Recently reported from your network. Please wait a moment.'
      status.value = 'error'
      const remaining = remainingLockSeconds()
      startCountdown(remaining > 0 ? remaining : RATE_LIMIT_SECONDS)
      return
    }
    if (!resp.ok) throw new Error(`report ${resp.status}`)

    markSentLocally()
    startCountdown(RATE_LIMIT_SECONDS)
    status.value = 'sent'
    description.value = ''
  } catch {
    if (seq !== requestSeq || !props.visible) return
    errorText.value = 'Failed to send. Please try again.'
    status.value = 'error'
  }
}
</script>

<style scoped>
.report-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100dvh;
  background-color: rgba(0, 0, 0, 0.5);
  z-index: 1100;
  display: flex;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;
}

.report-panel {
  width: 400px;
  max-width: 90vw;
  background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  overflow: hidden;
  animation: reportPopIn 0.25s ease-out;
}

@keyframes reportPopIn {
  from {
    transform: scale(0.92);
    opacity: 0;
  }
  to {
    transform: scale(1);
    opacity: 1;
  }
}

.report-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-color, #e2e8f0);
  background: linear-gradient(135deg, var(--bg-color, #ffffff) 0%, rgba(0, 173, 181, 0.05) 100%);
}

.report-header h3 {
  margin: 0;
  font-size: 1.2rem;
  color: var(--text-color, #2c3e50);
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
}

.report-title-icon {
  color: var(--primary-color, #00adb5);
  display: flex;
  align-items: center;
}

.report-close-btn {
  background: none;
  border: none;
  cursor: pointer;
  padding: 8px;
  border-radius: 8px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-color, #2c3e50);
}

.report-close-btn:hover {
  background-color: var(--hover-bg, #f7fafc);
  transform: scale(1.1);
}

.report-body {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.report-hint-text {
  margin: 0;
  font-size: 0.95rem;
  color: var(--text-color, #2c3e50);
  line-height: 1.5;
}

.report-hint-sub {
  font-size: 0.8rem;
  color: #94a3b8;
}

.report-textarea {
  width: 100%;
  box-sizing: border-box;
  border: 2px solid var(--border-color, #e2e8f0);
  border-radius: 12px;
  padding: 12px 14px;
  font-size: 0.95rem;
  font-family: inherit;
  color: var(--text-color, #2c3e50);
  background-color: #ffffff;
  resize: vertical;
  min-height: 90px;
  transition: border-color 0.2s ease;
}

.report-textarea:focus {
  outline: none;
  border-color: var(--primary-color, #00adb5);
}

.report-textarea:disabled {
  opacity: 0.6;
}

.report-char-count {
  text-align: right;
  font-size: 0.75rem;
  color: #94a3b8;
  margin-top: -8px;
}

.report-success {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 10px 0 4px;
}

.report-success-icon {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--primary-color, #00adb5) 0%, var(--primary-hover, #00c4cc) 100%);
  color: white;
  font-size: 1.6rem;
  display: flex;
  align-items: center;
  justify-content: center;
  animation: reportPopIn 0.3s ease-out;
}

.report-success p {
  margin: 0;
  font-size: 1rem;
  font-weight: 500;
  color: var(--text-color, #2c3e50);
}

.report-error {
  background-color: rgba(243, 129, 129, 0.12);
  border: 1px solid rgba(243, 129, 129, 0.4);
  color: #d9534f;
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 0.88rem;
}

.report-send-btn {
  border: none;
  border-radius: 12px;
  padding: 12px 20px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  color: white;
  background: linear-gradient(135deg, var(--primary-color, #00adb5) 0%, var(--primary-hover, #00c4cc) 100%);
  box-shadow: 0 2px 8px rgba(0, 173, 181, 0.3);
  transition: all 0.2s ease;
}

.report-send-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(0, 173, 181, 0.4);
}

.report-send-btn:active:not(:disabled) {
  transform: scale(0.97);
}

.report-send-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
  box-shadow: none;
}

.report-countdown-hint {
  text-align: center;
  font-size: 0.8rem;
  color: #94a3b8;
  margin-top: -6px;
}

/* 深色主题 */
.dark .report-panel {
  background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
  border-color: rgba(255, 255, 255, 0.1);
}

.dark .report-header {
  background: linear-gradient(135deg, #1a1a1a 0%, rgba(0, 173, 181, 0.05) 100%);
  border-bottom-color: #404040;
}

.dark .report-header h3,
.dark .report-hint-text,
.dark .report-success p {
  color: #e0e0e0;
}

.dark .report-close-btn {
  color: #e0e0e0;
}

.dark .report-close-btn:hover {
  background-color: #404040;
}

.dark .report-textarea {
  background-color: #2d2d2d;
  border-color: #404040;
  color: #e0e0e0;
}
</style>
