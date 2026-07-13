<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'
import type { CartState } from '../types'
import { STATE_LABELS } from '../types'

defineProps<{
  capturing: boolean
}>()

const counts = ref<Record<CartState, number>>({
  forward: 0,
  stop: 0,
  left: 0,
  right: 0,
  search: 0,
})

const total = computed(() => Object.values(counts.value).reduce((a, b) => a + b, 0))

let interval: ReturnType<typeof setInterval> | null = null

async function fetchStats() {
  try {
    const response = await fetch('/dataset/stats')
    if (!response.ok) return
    const data = await response.json()
    counts.value = data.counts
  } catch {
    // backend may not be running yet
  }
}

onMounted(() => {
  fetchStats()
  interval = setInterval(fetchStats, 500)
})

onUnmounted(() => {
  if (interval) clearInterval(interval)
})

defineExpose({ fetchStats })
</script>

<template>
  <div class="stats">
    <h3>
      Dataset samples
      <span v-if="capturing" class="recording">● recording</span>
    </h3>
    <p class="total">{{ total }} total</p>
    <ul>
      <li v-for="(count, name) in counts" :key="name">
        <span>{{ STATE_LABELS[name as CartState] }}</span>
        <strong>{{ count }}</strong>
      </li>
    </ul>
    <p class="hint">
      {{ capturing
        ? 'Sampling at 10 Hz for the latched state.'
        : 'Press Start capturing, then use W/A/S/D/Q to label.' }}
    </p>
  </div>
</template>

<style scoped>
.stats h3 {
  margin: 0 0 8px;
  font-size: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.recording {
  font-size: 11px;
  font-weight: 600;
  color: #ef4444;
  animation: pulse 1s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

.total {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--muted);
}

ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

li {
  display: flex;
  justify-content: space-between;
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: 14px;
}

.hint {
  margin: 12px 0 0;
  font-size: 12px;
  color: var(--muted);
}
</style>
