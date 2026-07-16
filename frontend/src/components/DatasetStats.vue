<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'
import type { CartState, DatasetSplit } from '../types'
import { EMPTY_COUNTS, STATE_LABELS } from '../types'

const props = defineProps<{
  capturing: boolean
  activeSplit: DatasetSplit
}>()

const emit = defineEmits<{
  'update:activeSplit': [split: DatasetSplit]
}>()

const trainCounts = ref<Record<CartState, number>>({ ...EMPTY_COUNTS })
const testCounts = ref<Record<CartState, number>>({ ...EMPTY_COUNTS })

const activeCounts = computed(() =>
  props.activeSplit === 'train' ? trainCounts.value : testCounts.value,
)
const trainTotal = computed(() =>
  Object.values(trainCounts.value).reduce((a, b) => a + b, 0),
)
const testTotal = computed(() =>
  Object.values(testCounts.value).reduce((a, b) => a + b, 0),
)
const activeTotal = computed(() =>
  Object.values(activeCounts.value).reduce((a, b) => a + b, 0),
)

let interval: ReturnType<typeof setInterval> | null = null

async function fetchStats() {
  try {
    const response = await fetch('/dataset/stats')
    if (!response.ok) return
    const data = await response.json()
    if (data.counts?.train) trainCounts.value = data.counts.train
    if (data.counts?.test) testCounts.value = data.counts.test
  } catch {
    // backend may not be running yet
  }
}

async function setSplit(split: DatasetSplit) {
  try {
    const response = await fetch('/dataset/split', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ split }),
    })
    if (!response.ok) return
    emit('update:activeSplit', split)
  } catch {
    // ignore
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

    <div class="split-toggle">
      <button
        type="button"
        :class="{ active: activeSplit === 'train' }"
        @click="setSplit('train')"
      >
        Train ({{ trainTotal }})
      </button>
      <button
        type="button"
        :class="{ active: activeSplit === 'test' }"
        @click="setSplit('test')"
      >
        Test ({{ testTotal }})
      </button>
    </div>

    <p class="total">
      Saving into <strong>{{ activeSplit }}</strong> · {{ activeTotal }} samples
    </p>
    <ul>
      <li v-for="(count, name) in activeCounts" :key="name">
        <span>{{ STATE_LABELS[name as CartState] }}</span>
        <strong>{{ count }}</strong>
      </li>
    </ul>
    <p class="hint">
      {{ capturing
        ? `Sampling at 10 Hz into ${activeSplit}/.`
        : 'Toggle Train vs Test, then Start capturing. Keep test data held out.' }}
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

.split-toggle {
  display: flex;
  gap: 4px;
  background: var(--code-bg);
  padding: 4px;
  border-radius: 8px;
  margin-bottom: 12px;
}

.split-toggle button {
  flex: 1;
  border: none;
  background: transparent;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  color: var(--text);
}

.split-toggle button.active {
  background: var(--bg);
  color: var(--text-h);
  box-shadow: var(--shadow);
  font-weight: 600;
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
