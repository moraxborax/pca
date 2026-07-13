<script setup lang="ts">
import { onMounted, ref } from 'vue'
import type { CartState } from '../types'
import { STATE_LABELS } from '../types'

const emit = defineEmits<{
  trained: []
}>()

const counts = ref<Record<CartState, number>>({
  forward: 0,
  stop: 0,
  left: 0,
  right: 0,
  search: 0,
})
const training = ref(false)
const message = ref('')
const error = ref('')

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

async function startTraining() {
  training.value = true
  message.value = ''
  error.value = ''
  try {
    const response = await fetch('/train', { method: 'POST' })
    const data = await response.json()
    if (!response.ok) {
      error.value = data.detail ?? 'Training failed'
      return
    }
    message.value = data.message ?? 'Training complete'
    emit('trained')
  } catch {
    error.value = 'Backend not reachable'
  } finally {
    training.value = false
  }
}

onMounted(fetchStats)
</script>

<template>
  <div class="train-panel">
    <h3>Train model</h3>
    <p class="hint">Runs PCA + MLP on the collected dataset and saves artifacts for Infer.</p>

    <ul>
      <li v-for="(count, name) in counts" :key="name">
        <span>{{ STATE_LABELS[name as CartState] }}</span>
        <strong>{{ count }}</strong>
      </li>
    </ul>

    <button class="train-btn" :disabled="training" @click="startTraining">
      {{ training ? 'Training…' : 'Start training' }}
    </button>

    <p v-if="message" class="ok">{{ message }}</p>
    <p v-if="error" class="err">{{ error }}</p>
  </div>
</template>

<style scoped>
.train-panel h3 {
  margin: 0 0 8px;
  font-size: 16px;
}

.hint {
  margin: 0 0 16px;
  font-size: 12px;
  color: var(--muted);
}

ul {
  list-style: none;
  padding: 0;
  margin: 0 0 16px;
}

li {
  display: flex;
  justify-content: space-between;
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: 14px;
}

.train-btn {
  width: 100%;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text-h);
  padding: 12px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 15px;
  font-weight: 600;
}

.train-btn:disabled {
  opacity: 0.6;
  cursor: wait;
}

.ok {
  margin: 12px 0 0;
  font-size: 13px;
  color: #22c55e;
}

.err {
  margin: 12px 0 0;
  font-size: 13px;
  color: #ef4444;
}
</style>
