<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import type { PredictionMessage } from '../types'
import { STATE_LABELS } from '../types'

const prediction = ref<PredictionMessage | null>(null)
const connected = ref(false)
let ws: WebSocket | null = null

function connect() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  ws = new WebSocket(`${protocol}//${window.location.host}/ws/predict`)

  ws.onopen = () => {
    connected.value = true
  }

  ws.onmessage = (event) => {
    prediction.value = JSON.parse(event.data) as PredictionMessage
  }

  ws.onclose = () => {
    connected.value = false
    setTimeout(connect, 2000)
  }

  ws.onerror = () => {
    ws?.close()
  }
}

onMounted(() => {
  connect()
})

onUnmounted(() => {
  ws?.close()
})
</script>

<template>
  <div class="prediction">
    <h3>ML prediction <span class="dot" :class="{ on: connected }" /></h3>
    <template v-if="prediction">
      <p class="pred-class" :class="prediction.class_name">
        {{ STATE_LABELS[prediction.class_name] }}
      </p>
      <div class="confidence-bar">
        <div class="fill" :style="{ width: `${prediction.confidence * 100}%` }" />
      </div>
      <p class="confidence-text">{{ (prediction.confidence * 100).toFixed(1) }}% confidence</p>
      <p class="commanded">Commanded: {{ STATE_LABELS[prediction.commanded_state] }}</p>
    </template>
    <p v-else class="waiting">Waiting for predictions…</p>
  </div>
</template>

<style scoped>
.prediction h3 {
  margin: 0 0 12px;
  font-size: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ef4444;
  display: inline-block;
}

.dot.on {
  background: #22c55e;
}

.pred-class {
  font-size: 36px;
  font-weight: 700;
  margin: 0 0 12px;
}

.pred-class.forward { color: #22c55e; }
.pred-class.backward { color: #14b8a6; }
.pred-class.stop { color: #ef4444; }
.pred-class.left { color: #3b82f6; }
.pred-class.right { color: #f59e0b; }
.pred-class.search { color: #a855f7; }

.confidence-bar {
  height: 8px;
  background: var(--border);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 6px;
}

.fill {
  height: 100%;
  background: var(--accent);
  transition: width 0.15s;
}

.confidence-text,
.commanded,
.waiting {
  margin: 0;
  font-size: 13px;
  color: var(--muted);
}

.commanded {
  margin-top: 8px;
}
</style>
