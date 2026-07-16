<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import MetricChart from './MetricChart.vue'
import type { CartState } from '../types'
import { EMPTY_COUNTS, STATE_LABELS } from '../types'

interface TrainHistory {
  epochs: number[]
  train_loss: number[]
  train_acc: number[]
  val_loss: number[]
  val_acc: number[]
  n_samples?: number
  n_train_samples?: number
  n_test_samples?: number
  residual_threshold?: number
  epochs_requested?: number
  steps_per_epoch?: number
  total_steps?: number
  batch_size?: number
  test_loss?: number | null
  test_acc?: number | null
  test_per_class?: Record<string, number> | null
}

interface TrainProgress {
  active: boolean
  epoch: number
  total_epochs: number
  step: number
  steps_per_epoch: number
  total_steps: number
  global_step: number
  message: string
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

const trainCounts = ref<Record<CartState, number>>({ ...EMPTY_COUNTS })
const testCounts = ref<Record<CartState, number>>({ ...EMPTY_COUNTS })
const training = ref(false)
const message = ref('')
const error = ref('')
const history = ref<TrainHistory | null>(null)
const pca = ref<PcaViz | null>(null)
const epochCount = ref(30)
const progress = ref<TrainProgress | null>(null)
let progressTimer: ReturnType<typeof setInterval> | null = null

const MIN_EPOCHS = 5
const MAX_EPOCHS = 100
const EPOCH_STEP = 5

const historySummary = computed(() => {
  if (!history.value?.epochs?.length) return null
  const h = history.value
  const epochsDone = h.epochs.length
  const stepsPerEpoch = h.steps_per_epoch ?? 0
  const totalSteps = h.total_steps ?? epochsDone * stepsPerEpoch
  return {
    epochsDone,
    epochsRequested: h.epochs_requested ?? epochsDone,
    stepsPerEpoch,
    totalSteps,
    batchSize: h.batch_size ?? 32,
  }
})

const progressPercent = computed(() => {
  if (!progress.value?.total_steps) return 0
  return Math.min(
    100,
    Math.round((progress.value.global_step / progress.value.total_steps) * 100),
  )
})

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

async function fetchProgress() {
  try {
    const response = await fetch('/train/progress')
    if (!response.ok) return
    progress.value = await response.json()
  } catch {
    // ignore while backend is busy
  }
}

function startProgressPolling() {
  stopProgressPolling()
  void fetchProgress()
  progressTimer = setInterval(() => {
    void fetchProgress()
  }, 400)
}

function stopProgressPolling() {
  if (progressTimer !== null) {
    clearInterval(progressTimer)
    progressTimer = null
  }
}

async function refreshViz() {
  await Promise.all([fetchMetrics(), fetchPca()])
}

async function startTraining() {
  training.value = true
  message.value = ''
  error.value = ''
  startProgressPolling()
  try {
    const response = await fetch('/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: epochCount.value }),
    })
    const data = await response.json()
    if (!response.ok) {
      error.value = data.detail ?? 'Training failed'
      return
    }
    message.value = data.message ?? 'Training complete'
    if (data.history) {
      history.value = data.history
    }
    await fetchProgress()
    await fetchPca()
    emit('trained')
  } catch {
    error.value = 'Backend not reachable'
  } finally {
    training.value = false
    stopProgressPolling()
  }
}

onMounted(async () => {
  await fetchStats()
  await refreshViz()
  await fetchProgress()
})

onUnmounted(() => {
  stopProgressPolling()
})
</script>

<template>
  <div class="train-panel">
    <h3>Train model</h3>
    <p class="hint">
      Fits PCA + MLP on <strong>train</strong> only. Held-out <strong>test</strong> is scored once at the end.
    </p>

    <div class="split-counts">
      <div>
        <p class="split-title">Train</p>
        <ul>
          <li v-for="(count, name) in trainCounts" :key="`tr-${name}`">
            <span>{{ STATE_LABELS[name as CartState] }}</span>
            <strong>{{ count }}</strong>
          </li>
        </ul>
      </div>
      <div>
        <p class="split-title">Test (held out)</p>
        <ul>
          <li v-for="(count, name) in testCounts" :key="`te-${name}`">
            <span>{{ STATE_LABELS[name as CartState] }}</span>
            <strong>{{ count }}</strong>
          </li>
        </ul>
      </div>
    </div>

    <label class="epoch-control">
      <span class="epoch-label">Epochs to train</span>
      <strong>{{ epochCount }}</strong>
      <input
        v-model.number="epochCount"
        type="range"
        :min="MIN_EPOCHS"
        :max="MAX_EPOCHS"
        :step="EPOCH_STEP"
        :disabled="training"
      />
      <span class="epoch-range">{{ MIN_EPOCHS }}–{{ MAX_EPOCHS }}</span>
    </label>

    <button class="train-btn" :disabled="training" @click="startTraining">
      {{ training ? 'Training…' : 'Start training' }}
    </button>

    <section v-if="training || progress?.message" class="progress-block">
      <div v-if="training || progress?.active" class="progress-bar">
        <div class="progress-fill" :style="{ width: `${progressPercent}%` }" />
      </div>
      <p v-if="progress" class="progress-text">
        <template v-if="progress.total_epochs">
          Epoch {{ progress.epoch }}/{{ progress.total_epochs }}
          · step {{ progress.step }}/{{ progress.steps_per_epoch }}
          · total {{ progress.global_step }}/{{ progress.total_steps }}
        </template>
        <template v-else-if="historySummary">
          Last run: {{ historySummary.epochsDone }}/{{ historySummary.epochsRequested }}
          epochs · {{ historySummary.totalSteps }} steps
        </template>
      </p>
      <p v-if="progress?.message" class="meta">{{ progress.message }}</p>
    </section>

    <p v-if="historySummary && !training" class="meta history-meta">
      Last trained {{ historySummary.epochsDone }} epochs
      ({{ historySummary.totalSteps }} steps, {{ historySummary.stepsPerEpoch }}/epoch,
      batch {{ historySummary.batchSize }})
    </p>

    <p v-if="message" class="ok">{{ message }}</p>
    <p v-if="error" class="err">{{ error }}</p>

    <section
      v-if="history && (history.test_acc != null || history.n_test_samples === 0)"
      class="block"
    >
      <h4>Held-out test</h4>
      <template v-if="history.test_acc != null">
        <p class="test-score">
          Accuracy {{ (history.test_acc * 100).toFixed(1) }}%
          · loss {{ history.test_loss?.toFixed(4) }}
          · {{ history.n_test_samples }} samples
        </p>
        <ul v-if="history.test_per_class">
          <li v-for="(acc, name) in history.test_per_class" :key="name">
            <span>{{ STATE_LABELS[name as CartState] ?? name }}</span>
            <strong>{{ (acc * 100).toFixed(1) }}%</strong>
          </li>
        </ul>
      </template>
      <p v-else class="meta">No test samples yet — capture with Dataset = Test.</p>
    </section>

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

.split-counts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 16px;
}

.split-title {
  margin: 0 0 6px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-h);
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

.epoch-control {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 4px 12px;
  align-items: center;
  margin-bottom: 16px;
  font-size: 13px;
}

.epoch-label {
  color: var(--text-h);
}

.epoch-control input[type='range'] {
  grid-column: 1 / -1;
  width: 100%;
}

.epoch-range {
  grid-column: 1 / -1;
  font-size: 11px;
  color: var(--muted);
}

.progress-block {
  margin-top: 12px;
}

.progress-bar {
  height: 8px;
  border-radius: 999px;
  background: var(--border);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #22c55e);
  transition: width 0.25s ease;
}

.progress-text {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--text-h);
}

.history-meta {
  margin-top: 8px;
}

.ok {
  margin: 12px 0 0;
  font-size: 13px;
  color: #22c55e;
}

.test-score {
  margin: 0 0 8px;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-h);
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
