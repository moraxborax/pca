<script setup lang="ts">
import { onMounted, ref } from 'vue'
import MetricChart from './MetricChart.vue'
import type { CartState } from '../types'
import { STATE_LABELS } from '../types'

interface TrainHistory {
  epochs: number[]
  train_loss: number[]
  train_acc: number[]
  val_loss: number[]
  val_acc: number[]
  n_samples?: number
  residual_threshold?: number
}

interface PcaViz {
  mean: string
  eigenimages: { index: number; variance_ratio: number; image: string }[]
  variance: { index: number; ratio: number; cumulative: number }[]
  reconstructions: {
    class_name: CartState
    original: string
    reconstructed: string
    residual_map: string
    residual: number
  }[]
  residual_threshold?: number
  n_components?: number
}

const emit = defineEmits<{
  trained: []
}>()

const counts = ref<Record<CartState, number>>({
  forward: 0,
  backward: 0,
  stop: 0,
  left: 0,
  right: 0,
  search: 0,
})
const training = ref(false)
const message = ref('')
const error = ref('')
const history = ref<TrainHistory | null>(null)
const pca = ref<PcaViz | null>(null)

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

async function fetchMetrics() {
  try {
    const response = await fetch('/train/metrics')
    if (!response.ok) {
      history.value = null
      return
    }
    history.value = await response.json()
  } catch {
    history.value = null
  }
}

async function fetchPca() {
  try {
    const response = await fetch('/pca/viz')
    if (!response.ok) {
      pca.value = null
      return
    }
    pca.value = await response.json()
  } catch {
    pca.value = null
  }
}

async function refreshViz() {
  await Promise.all([fetchMetrics(), fetchPca()])
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
    if (data.history) {
      history.value = data.history
    }
    await fetchPca()
    emit('trained')
  } catch {
    error.value = 'Backend not reachable'
  } finally {
    training.value = false
  }
}

onMounted(async () => {
  await fetchStats()
  await refreshViz()
})
</script>

<template>
  <div class="train-panel">
    <h3>Train model</h3>
    <p class="hint">Runs PCA + MLP, then shows curves and PCA views (TensorBoard-style).</p>

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

    <section v-if="history?.epochs?.length" class="block">
      <h4>Training curves</h4>
      <MetricChart
        title="Loss"
        :series="[
          { name: 'train', color: '#3b82f6', values: history.train_loss },
          { name: 'val', color: '#f59e0b', values: history.val_loss },
        ]"
      />
      <MetricChart
        title="Accuracy"
        :y-min="0"
        :y-max="1"
        :series="[
          { name: 'train', color: '#22c55e', values: history.train_acc },
          { name: 'val', color: '#a855f7', values: history.val_acc },
        ]"
      />
      <p v-if="history.residual_threshold != null" class="meta">
        residual gate threshold: {{ history.residual_threshold.toFixed(3) }}
      </p>
    </section>

    <section v-if="pca" class="block">
      <h4>PCA</h4>
      <p class="meta">{{ pca.n_components }} components · mean image</p>
      <img class="thumb" :src="pca.mean" alt="PCA mean" />

      <p class="meta">Top eigenimages</p>
      <div class="eigen-grid">
        <figure v-for="e in pca.eigenimages" :key="e.index">
          <img :src="e.image" :alt="`PC${e.index}`" />
          <figcaption>PC{{ e.index }} · {{ (e.variance_ratio * 100).toFixed(1) }}%</figcaption>
        </figure>
      </div>

      <MetricChart
        v-if="pca.variance.length"
        title="Cumulative variance"
        :y-min="0"
        :y-max="1"
        :series="[
          {
            name: 'cumulative',
            color: '#aa3bff',
            values: pca.variance.map((v) => v.cumulative),
          },
        ]"
      />

      <p class="meta">Reconstruction (original · recon · |residual|)</p>
      <div v-for="r in pca.reconstructions" :key="r.class_name" class="recon-row">
        <span class="recon-label">{{ STATE_LABELS[r.class_name] }}</span>
        <img :src="r.original" alt="original" />
        <img :src="r.reconstructed" alt="reconstructed" />
        <img :src="r.residual_map" alt="residual" />
        <span class="recon-r">r={{ r.residual.toFixed(2) }}</span>
      </div>
    </section>

    <p v-else-if="!training" class="hint">Train once to populate metrics and PCA views.</p>
  </div>
</template>

<style scoped>
.train-panel h3 {
  margin: 0 0 8px;
  font-size: 16px;
}

.train-panel h4 {
  margin: 0 0 8px;
  font-size: 14px;
  color: var(--text-h);
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

.block {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
}

.meta {
  margin: 8px 0;
  font-size: 11px;
  color: var(--muted);
}

.thumb {
  width: 96px;
  height: 96px;
  image-rendering: pixelated;
  border-radius: 4px;
  background: #000;
}

.eigen-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-bottom: 8px;
}

figure {
  margin: 0;
  text-align: center;
}

figure img {
  width: 100%;
  aspect-ratio: 1;
  image-rendering: pixelated;
  border-radius: 4px;
  background: #000;
}

figcaption {
  font-size: 10px;
  color: var(--muted);
  margin-top: 2px;
}

.recon-row {
  display: grid;
  grid-template-columns: 64px repeat(3, 1fr) auto;
  gap: 4px;
  align-items: center;
  margin-bottom: 8px;
  font-size: 10px;
}

.recon-row img {
  width: 100%;
  aspect-ratio: 1;
  image-rendering: pixelated;
  border-radius: 2px;
  background: #000;
}

.recon-label {
  font-size: 10px;
  color: var(--text-h);
}

.recon-r {
  color: var(--muted);
  font-size: 10px;
}
</style>
