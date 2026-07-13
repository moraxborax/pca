<script setup lang="ts">
import { ref } from 'vue'
import DatasetStats from './components/DatasetStats.vue'
import PredictionPanel from './components/PredictionPanel.vue'
import StateDisplay from './components/StateDisplay.vue'
import TrainPanel from './components/TrainPanel.vue'
import VideoStream from './components/VideoStream.vue'
import { useKeyboard } from './composables/useKeyboard'
import type { AppMode, CartState } from './types'

const mode = ref<AppMode>('label')
const capturing = ref(false)
const statsRef = ref<InstanceType<typeof DatasetStats> | null>(null)
const error = ref('')

async function postState(state: CartState) {
  if (mode.value !== 'label' && mode.value !== 'infer') return
  error.value = ''
  try {
    const response = await fetch('/state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state }),
    })
    if (!response.ok) {
      const data = await response.json()
      error.value = data.detail ?? 'Failed to set state'
    }
    statsRef.value?.fetchStats()
  } catch {
    error.value = 'Backend not reachable'
  }
}

const { currentState } = useKeyboard(postState)

async function setMode(newMode: AppMode) {
  error.value = ''
  try {
    const response = await fetch('/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: newMode }),
    })
    if (!response.ok) {
      const data = await response.json()
      error.value = data.detail ?? 'Failed to switch mode'
      return
    }
    mode.value = newMode
    if (newMode !== 'label') {
      capturing.value = false
    }
  } catch {
    error.value = 'Backend not reachable'
  }
}

async function toggleCapture() {
  error.value = ''
  const next = !capturing.value
  try {
    const response = await fetch('/capture', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ capturing: next }),
    })
    if (!response.ok) {
      const data = await response.json()
      error.value = data.detail ?? 'Failed to toggle capture'
      return
    }
    capturing.value = next
  } catch {
    error.value = 'Backend not reachable'
  }
}

async function clearDataset() {
  if (!window.confirm('Delete all captured samples? This cannot be undone.')) {
    return
  }
  error.value = ''
  try {
    if (capturing.value) {
      await fetch('/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ capturing: false }),
      })
      capturing.value = false
    }
    const response = await fetch('/dataset', { method: 'DELETE' })
    if (!response.ok) {
      const data = await response.json()
      error.value = data.detail ?? 'Failed to clear dataset'
      return
    }
    statsRef.value?.fetchStats()
  } catch {
    error.value = 'Backend not reachable'
  }
}
</script>

<template>
  <div class="app" tabindex="0">
    <header>
      <h1>Robot Cart</h1>
      <div class="mode-toggle">
        <button :class="{ active: mode === 'label' }" @click="setMode('label')">Capture</button>
        <button :class="{ active: mode === 'train' }" @click="setMode('train')">Train</button>
        <button :class="{ active: mode === 'infer' }" @click="setMode('infer')">Infer</button>
      </div>
    </header>

    <p v-if="error" class="error">{{ error }}</p>

    <main>
      <VideoStream />
      <aside>
        <StateDisplay v-if="mode !== 'train'" :current-state="currentState" />
        <template v-if="mode === 'label'">
          <button
            class="capture-btn"
            :class="{ recording: capturing }"
            @click="toggleCapture"
          >
            {{ capturing ? 'Stop capturing' : 'Start capturing' }}
          </button>
          <DatasetStats ref="statsRef" :capturing="capturing" />
          <button class="clear-btn" @click="clearDataset">Clear dataset</button>
        </template>
        <TrainPanel v-else-if="mode === 'train'" />
        <PredictionPanel v-else />
      </aside>
    </main>
  </div>
</template>

<style scoped>
.app {
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px;
  outline: none;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

h1 {
  margin: 0;
  font-size: 24px;
}

.mode-toggle {
  display: flex;
  gap: 4px;
  background: var(--code-bg);
  padding: 4px;
  border-radius: 8px;
}

.mode-toggle button {
  border: none;
  background: transparent;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  color: var(--text);
}

.mode-toggle button.active {
  background: var(--bg);
  color: var(--text-h);
  box-shadow: var(--shadow);
}

.capture-btn {
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text-h);
  padding: 12px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 15px;
  font-weight: 600;
}

.capture-btn.recording {
  border-color: #ef4444;
  background: rgba(239, 68, 68, 0.12);
  color: #ef4444;
}

.clear-btn {
  border: 1px solid var(--border);
  background: transparent;
  color: var(--muted);
  padding: 8px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
}

.clear-btn:hover {
  border-color: #ef4444;
  color: #ef4444;
}

.error {
  color: #ef4444;
  background: rgba(239, 68, 68, 0.1);
  padding: 8px 12px;
  border-radius: 6px;
  margin: 0 0 16px;
  font-size: 14px;
}

main {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 24px;
}

aside {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 20px;
  border: 1px solid var(--border);
  border-radius: 8px;
}

@media (max-width: 800px) {
  main {
    grid-template-columns: 1fr;
  }
}
</style>
