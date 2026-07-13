<script setup lang="ts">
import type { CartState } from '../types'
import { STATE_LABELS } from '../types'

defineProps<{
  currentState: CartState
}>()

const legend = [
  { key: 'W', state: 'forward' as CartState },
  { key: 'S', state: 'stop' as CartState },
  { key: 'A', state: 'left' as CartState },
  { key: 'D', state: 'right' as CartState },
  { key: 'Q', state: 'search' as CartState },
]
</script>

<template>
  <div class="state-display">
    <p class="label">Current state</p>
    <p class="state" :class="currentState">{{ STATE_LABELS[currentState] }}</p>
    <div class="legend">
      <div v-for="item in legend" :key="item.key" class="legend-item" :class="{ active: currentState === item.state }">
        <kbd>{{ item.key }}</kbd>
        <span>{{ STATE_LABELS[item.state] }}</span>
      </div>
    </div>
    <p class="hint">Keys latch on press — release does not revert.</p>
  </div>
</template>

<style scoped>
.state-display {
  text-align: center;
}

.label {
  margin: 0 0 8px;
  color: var(--muted);
  font-size: 14px;
}

.state {
  font-size: 48px;
  font-weight: 700;
  margin: 0 0 24px;
  letter-spacing: 2px;
}

.state.forward { color: #22c55e; }
.state.stop { color: #ef4444; }
.state.left { color: #3b82f6; }
.state.right { color: #f59e0b; }
.state.search { color: #a855f7; }

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-bottom: 12px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid var(--border);
  font-size: 13px;
}

.legend-item.active {
  border-color: var(--accent);
  background: var(--accent-bg);
}

kbd {
  font-family: var(--mono);
  background: var(--code-bg);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}

.hint {
  margin: 0;
  font-size: 12px;
  color: var(--muted);
}
</style>
