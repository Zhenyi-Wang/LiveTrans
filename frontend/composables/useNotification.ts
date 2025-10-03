import { ref, createApp } from 'vue'
import Notification from '../components/common/Notification.vue'

interface NotificationOptions {
  message: string
  type?: 'info' | 'success' | 'warning' | 'error'
  duration?: number
  closable?: boolean
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center' | 'bottom-center'
  customStyle?: Record<string, string>
}

const notifications = ref<Array<{
  id: string
  component: any
  container: HTMLElement
}>>([])

export function useNotification() {
  const show = (options: NotificationOptions) => {
    if (typeof window === 'undefined') return

    const id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`

    // 创建容器
    const container = document.createElement('div')
    container.id = id
    document.body.appendChild(container)

    // 创建Vue应用实例
    const app = createApp(Notification, {
      ...options,
      onClose: () => {
        close(id)
      }
    })

    const instance = app.mount(container)

    // 存储通知实例
    notifications.value.push({
      id,
      component: instance,
      container
    })

    return id
  }

  const close = (id: string) => {
    const index = notifications.value.findIndex(n => n.id === id)
    if (index !== -1) {
      const notification = notifications.value[index]

      // 等待动画完成后移除DOM
      setTimeout(() => {
        if (notification.container.parentNode) {
          notification.container.parentNode.removeChild(notification.container)
        }
      }, 300)

      notifications.value.splice(index, 1)
    }
  }

  const closeAll = () => {
    notifications.value.forEach(notification => {
      if (notification.container.parentNode) {
        notification.container.parentNode.removeChild(notification.container)
      }
    })
    notifications.value = []
  }

  // 便捷方法
  const info = (message: string, options?: Omit<NotificationOptions, 'message' | 'type'>) => {
    return show({ message, type: 'info', ...options })
  }

  const success = (message: string, options?: Omit<NotificationOptions, 'message' | 'type'>) => {
    return show({ message, type: 'success', ...options })
  }

  const warning = (message: string, options?: Omit<NotificationOptions, 'message' | 'type'>) => {
    return show({ message, type: 'warning', ...options })
  }

  const error = (message: string, options?: Omit<NotificationOptions, 'message' | 'type'>) => {
    return show({ message, type: 'error', ...options })
  }

  return {
    show,
    close,
    closeAll,
    info,
    success,
    warning,
    error
  }
}

// 全局单例
let globalNotification: ReturnType<typeof useNotification> | null = null

export function useGlobalNotification() {
  if (!globalNotification) {
    globalNotification = useNotification()
  }
  return globalNotification
}