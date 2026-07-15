<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  title: string
  series: { name: string; color: string; values: number[] }[]
  yMin?: number
  yMax?: number
}>()

const width = 280
const height = 120
const pad = { top: 12, right: 8, bottom: 20, left: 36 }

const plotW = width - pad.left - pad.right
const plotH = height - pad.top - pad.bottom

const bounds = computed(() => {
  const all = props.series.flatMap((s) => s.values)
  if (!all.length) return { min: 0, max: 1 }
  let min = props.yMin ?? Math.min(...all)
  let max = props.yMax ?? Math.max(...all)
  if (min === max) {
    min -= 0.05
    max += 0.05
  }
  return { min, max }
})

function xAt(i: number, n: number) {
  if (n <= 1) return pad.left + plotW / 2
  return pad.left + (i / (n - 1)) * plotW
}

function yAt(v: number) {
  const { min, max } = bounds.value
  return pad.top + plotH - ((v - min) / (max - min)) * plotH
}

function pathFor(values: number[]) {
  if (!values.length) return ''
  return values
    .map((v, i) => `${i === 0 ? 'M' : 'L'} ${xAt(i, values.length).toFixed(1)} ${yAt(v).toFixed(1)}`)
    .join(' ')
}

const yTicks = computed(() => {
  const { min, max } = bounds.value
  return [min, (min + max) / 2, max]
})
</script>

<template>
  <div class="chart">
    <p class="title">{{ title }}</p>
    <svg :viewBox="`0 0 ${width} ${height}`" role="img">
      <line
        v-for="(t, i) in yTicks"
        :key="i"
        :x1="pad.left"
        :x2="width - pad.right"
        :y1="yAt(t)"
        :y2="yAt(t)"
        class="grid"
      />
      <text
        v-for="(t, i) in yTicks"
        :key="`l${i}`"
        :x="pad.left - 4"
        :y="yAt(t) + 3"
        class="tick"
        text-anchor="end"
      >
        {{ t.toFixed(2) }}
      </text>
      <path
        v-for="s in series"
        :key="s.name"
        :d="pathFor(s.values)"
        fill="none"
        :stroke="s.color"
        stroke-width="2"
        stroke-linejoin="round"
        stroke-linecap="round"
      />
    </svg>
    <div class="legend">
      <span v-for="s in series" :key="s.name" class="item">
        <i :style="{ background: s.color }" />
        {{ s.name }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.chart {
  margin-top: 12px;
}

.title {
  margin: 0 0 4px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-h);
}

svg {
  width: 100%;
  height: auto;
  display: block;
  background: var(--code-bg);
  border-radius: 6px;
}

.grid {
  stroke: var(--border);
  stroke-width: 1;
}

.tick {
  fill: var(--muted);
  font-size: 9px;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 6px;
  font-size: 11px;
  color: var(--muted);
}

.item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

i {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  display: inline-block;
}
</style>
