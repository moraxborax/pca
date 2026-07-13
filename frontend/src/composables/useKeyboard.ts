import { onMounted, onUnmounted, ref } from 'vue'
import type { CartState } from '../types'
import { KEY_TO_STATE } from '../types'

export function useKeyboard(onStateChange: (state: CartState) => void) {
  const currentState = ref<CartState>('stop')

  function handleKeydown(event: KeyboardEvent) {
    if (event.repeat) return
    const state = KEY_TO_STATE[event.key.toLowerCase()]
    if (!state) return
    event.preventDefault()
    currentState.value = state
    onStateChange(state)
  }

  onMounted(() => {
    window.addEventListener('keydown', handleKeydown)
  })

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown)
  })

  return { currentState }
}
