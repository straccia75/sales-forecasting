<!-- src/components/ForecastChart.vue -->
<template>
  <div class="fc-wrapper">
    <!-- Chips opcionales (internas) -->
    <div v-if="showChips && groupNames.length" class="fc-chips" role="toolbar" aria-label="Series filters">
      <button
        v-for="name in groupNames"
        :key="name"
        class="fc-chip"
        :class="{ 'is-active': selectedHas(name) }"
        type="button"
        @click="onChipClick(name)"
        :aria-pressed="selectedHas(name) ? 'true' : 'false'"
      >
        {{ name }}
      </button>

      <button
        v-if="selectMode === 'multiple'"
        key="__toggle_all"
        class="fc-chip util"
        type="button"
        @click="toggleAll()"
      >
        {{ internalSelected.size === groupNames.length ? 'Clear all' : 'Select all' }}
      </button>
    </div>

    <!-- Contenedor del chart -->
    <div ref="chartEl" class="fc-chart" role="img" :aria-label="ariaLabel"></div>

    <!-- Estado vacío -->
    <div v-if="!hasData" class="fc-empty">No data</div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'

/** ---------- Types esperados desde el backend ---------- */
type Point = {
  group: string | Record<string, any>
  kind: 'history' | 'forecast'
  date: string
  yhat: number | string
  yhat_lower?: number | string | null
  yhat_upper?: number | string | null
}
type ResponseShape = { forecast: Point[] }

/** ---------- Props ---------- */
interface ForecastChartProps {
  /** Puedes pasar el objeto de la API { forecast: [...] } */
  response?: ResponseShape | null
  /** O directamente el array de puntos */
  data?: Point[] | null

  showChips?: boolean
  selectMode?: 'single' | 'multiple'
  activeGroups?: string[]
  numberLocale?: string
  axisLabelFormatter?: (val: number) => string
  yAxisLabel?: string
  ariaLabel?: string
}

const props = withDefaults(defineProps<ForecastChartProps>(), {
  showChips: false,
  selectMode: 'multiple'
})

const emit = defineEmits<{ (e: 'update:activeGroups', value: string[]): void }>()

/** ---------- Estado ---------- */
const chartEl = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null
let ro: ResizeObserver | null = null

/** Fuente unificada de datos: response.forecast o data */
const rows = computed<Point[]>(() => {
  if (props.data && Array.isArray(props.data)) return props.data
  if (props.response?.forecast && Array.isArray(props.response.forecast)) return props.response.forecast
  return []
})

const hasData = computed(() => rows.value.length > 0)

const showChips = computed(() => props.showChips === true)
const selectMode = computed(() => props.selectMode ?? 'multiple')

/** ---------- Normalizadores ---------- */
const keyOf = (s: string) => s.trim().toLowerCase()
function keyOfGroup(g: Point['group']): string {
  if (typeof g === 'string') return g
  const entries = Object.entries(g ?? {}).sort(([a],[b]) => a.localeCompare(b))
  return entries.map(([k,v]) => `${k}:${String(v)}`).join('|')
}

/** Conversión robusta a número finito (acepta "1,23") */
function toFiniteNumber(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null
  if (typeof v === 'string') {
    const n = Number(v.replace(',', '.'))
    return Number.isFinite(n) ? n : null
  }
  return null
}

/** Parser de fecha (ISO u otros parseables por Date) */
function parseDateMs(d: string): number | null {
  const t = new Date(d).getTime()
  return Number.isFinite(t) ? t : null
}

/** ---------- Colección de grupos ---------- */
const groupNames = computed(() => {
  const set = new Set<string>()
  for (const p of rows.value) set.add(keyOfGroup(p.group))
  return Array.from(set).sort((a,b) => a.localeCompare(b))
})

/** Selección interna (KEYS) */
const internalSelected = ref<Set<string>>(new Set())

watch([groupNames, () => props.activeGroups], ([names, external]) => {
  if (external?.length) internalSelected.value = new Set(external.map(keyOf))
  else internalSelected.value = new Set(names.map(keyOf))
  syncLegendToSelection()
})

const selectedHas = (name: string) => internalSelected.value.has(keyOf(name))

/** ---------- Interacciones de chips ---------- */
function onChipClick(name: string) {
  const k = keyOf(name)
  if (selectMode.value === 'single') {
    const already = internalSelected.value.has(k)
    internalSelected.value = already ? new Set() : new Set([k])
  } else {
    const next = new Set(internalSelected.value)
    if (next.has(k)) next.delete(k)
    else next.add(k)
    internalSelected.value = next
  }
  emit('update:activeGroups', Array.from(internalSelected.value))
  syncLegendToSelection()
  render()
}

function toggleAll() {
  const allKeys = new Set(groupNames.value.map(keyOf))
  const next = (internalSelected.value.size === allKeys.size) ? new Set<string>() : allKeys
  internalSelected.value = next
  emit('update:activeGroups', Array.from(next))
  syncLegendToSelection()
  render()
}

/** ---------- Sincronía con la leyenda ---------- */
function syncLegendToSelection() {
  if (!chart) return
  const selected = internalSelected.value
  for (const n of groupNames.value) {
    const isOn = selected.has(keyOf(n))
    chart.dispatchAction({ type: isOn ? 'legendSelect' : 'legendUnSelect', name: n })
  }
}

/** ---------- Transformación: por grupo, ordenado, sin solape ---------- */
type SeriesPrepared = {
  name: string
  history: Array<[number, number]>
  forecast: Array<[number, number]>
  lower: Array<[number, number | null]>
  upper: Array<[number, number | null]>
  hasBand: boolean
}

const groupsPrepared = computed<Record<string, SeriesPrepared>>(() => {
  const byKey: Record<string, SeriesPrepared> = {}

  // Acumula
  for (const p of rows.value) {
    const name = keyOfGroup(p.group)
    const ts = parseDateMs(p.date)
    const y = toFiniteNumber(p.yhat)
    if (ts == null || y == null) continue

    if (!byKey[name]) byKey[name] = { name, history: [], forecast: [], lower: [], upper: [], hasBand: false }
    const entry = byKey[name]

    if (p.kind === 'history') {
      entry.history.push([ts, y])
    } else {
      entry.forecast.push([ts, y])
      const lo = toFiniteNumber(p.yhat_lower as any)
      const hi = toFiniteNumber(p.yhat_upper as any)
      entry.lower.push([ts, lo])
      entry.upper.push([ts, hi])
    }
  }

  // Orden y dedupe/solape
  for (const key of Object.keys(byKey)) {
    const s = byKey[key]
    const sortAsc = (a: [number, number|null], b: [number, number|null]) => a[0] - b[0]
    s.history.sort(sortAsc as any)
    s.forecast.sort(sortAsc as any)
    s.lower.sort(sortAsc as any)
    s.upper.sort(sortAsc as any)

    if (s.history.length && s.forecast.length) {
      const lastHistTs = s.history[s.history.length - 1][0]
      // Elimina forecast <= last history
      s.forecast = s.forecast.filter(([ts]) => ts > lastHistTs)
      s.lower   = s.lower.filter(([ts]) => ts > lastHistTs)
      s.upper   = s.upper.filter(([ts]) => ts > lastHistTs)
    }

    // ¿Banda utilizable?
    const totalF = s.forecast.length
    if (totalF > 0) {
      let ok = 0
      for (let i = 0; i < s.lower.length; i++) {
        const l = s.lower[i]?.[1]
        const u = s.upper[i]?.[1]
        if (typeof l === 'number' && Number.isFinite(l) && typeof u === 'number' && Number.isFinite(u) && l <= u) ok++
      }
      s.hasBand = ok / totalF >= 0.3
    }
  }

  return byKey
})

const hasAnySeries = computed(() =>
  Object.values(groupsPrepared.value).some(s => (s.history.length + s.forecast.length) > 0)
)

/** ---------- Formateo de números ---------- */
const nf = computed(() => {
  try {
    return new Intl.NumberFormat(props.numberLocale || undefined, { maximumFractionDigits: 0 })
  } catch {
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 })
  }
})

function formatAxis(val: number) {
  return props.axisLabelFormatter ? props.axisLabelFormatter(val) : nf.value.format(val)
}

/** ---------- ECharts Options ---------- */
function buildSeries(): echarts.SeriesOption[] {
  const out: echarts.SeriesOption[] = []

  for (const key of Object.keys(groupsPrepared.value)) {
    const s = groupsPrepared.value[key]
    if (!s.history.length && !s.forecast.length) continue

    // Banda (opcional)
    if (s.hasBand) {
      out.push({
        name: `${s.name} (upper)`,
        type: 'line',
        data: s.upper,
        showSymbol: false,
        silent: true,
        lineStyle: { width: 0 },
        areaStyle: { opacity: 0.12 },
        encode: { x: 0, y: 1 },
        z: 1
      } as echarts.SeriesOption)
      out.push({
        name: `${s.name} (lower)`,
        type: 'line',
        data: s.lower,
        showSymbol: false,
        silent: true,
        lineStyle: { width: 0 },
        areaStyle: { opacity: 0.12 },
        encode: { x: 0, y: 1 },
        z: 1
      } as echarts.SeriesOption)
    }

    // Historia
    out.push({
      name: s.name,
      type: 'line',
      data: s.history,
      showSymbol: false,
      sampling: 'lttb',
      progressive: 4000,
      emphasis: { focus: 'series' },
      lineStyle: { width: 2 },
      encode: { x: 0, y: 1 },
      z: 2
    } as echarts.SeriesOption)

    // Pronóstico
    if (s.forecast.length) {
      out.push({
        name: s.name,
        type: 'line',
        data: s.forecast,
        showSymbol: false,
        sampling: 'lttb',
        progressive: 4000,
        emphasis: { focus: 'series' },
        lineStyle: { width: 2, type: 'dashed' },
        encode: { x: 0, y: 1 },
        z: 3
      } as echarts.SeriesOption)
    }
  }
  return out
}

function buildOption(): echarts.EChartsOption {
  // Selección inicial por KEY
  const selectedMap: Record<string, boolean> = {}
  const selectedSet = internalSelected.value.size > 0
    ? internalSelected.value
    : new Set(groupNames.value.map(keyOf))
  for (const n of groupNames.value) selectedMap[n] = selectedSet.has(keyOf(n))

  const option: echarts.EChartsOption = {
    grid: { containLabel: true, left: 16, right: 16, top: 8, bottom: 64 },
    tooltip: {
      trigger: 'axis',
      confine: true,
      axisPointer: { type: 'line' },
      valueFormatter: (v: any) => (typeof v === 'number' ? nf.value.format(v) : String(v))
    },
    legend: {
      type: 'scroll',
      top: 0,
      selected: selectedMap
    },
    xAxis: {
      type: 'time',
      axisLabel: { hideOverlap: true },
      boundaryGap: false
    },
    yAxis: {
      type: 'value',
      name: props.yAxisLabel || undefined,
      nameLocation: 'end',
      nameGap: 10,
      axisLabel: { margin: 8, formatter: (val: number) => formatAxis(val) },
      splitLine: { show: true }
    },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0, throttle: 50 },
      { type: 'slider', xAxisIndex: 0, show: true, bottom: 8, height: 32, showDataShadow: true, brushSelect: false }
    ],
    series: buildSeries()
  }
  return option
}

/** ---------- Ciclo de vida ---------- */
function mountChart() {
  if (!chartEl.value) return
  chart = echarts.init(chartEl.value, undefined, { renderer: 'canvas' })

  if (internalSelected.value.size === 0 && groupNames.value.length > 0) {
    internalSelected.value = new Set(groupNames.value.map(keyOf))
  }

  render()

  ro = new ResizeObserver(() => chart?.resize())
  ro.observe(chartEl.value)

  attachLegendListener()
}

function disposeChart() {
  ro?.disconnect()
  ro = null
  if (chart) { chart.dispose(); chart = null }
}

function render() {
  if (!chart) return
  const option = buildOption()
  chart.setOption(option, { notMerge: true, lazyUpdate: false })
  nextTick(syncLegendToSelection)
}

function attachLegendListener() {
  if (!chart) return
  chart.on('legendselectchanged', (evt: any) => {
    const name = evt.name as string
    const k = keyOf(name)
    const isSelected = evt.selected?.[name] ?? true
    const next = new Set(internalSelected.value)
    if (isSelected) next.add(k)
    else next.delete(k)
    internalSelected.value = next
    emit('update:activeGroups', Array.from(next))
  })
}

onMounted(mountChart)
onBeforeUnmount(disposeChart)

/** Re-renders */
watch([rows, groupNames], () => { nextTick(render) })
watch(() => props.numberLocale, () => render())
watch(() => props.yAxisLabel, () => render())
watch(selectMode, () => render())
watch(() => props.activeGroups, (v) => {
  if (!v) return
  internalSelected.value = new Set(v.map(keyOf))
  syncLegendToSelection()
})

const ariaLabel = computed(() => props.ariaLabel ?? 'Sales forecast chart')
</script>

<style scoped>
.fc-wrapper {
  position: relative;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  -webkit-tap-highlight-color: transparent;
}

.fc-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
  align-items: center;
}

.fc-chip {
  border: 1px solid #cfd4dc;
  background: #fff;
  border-radius: 999px;
  padding: 6px 12px;
  font-size: 12px;
  line-height: 1.2;
  cursor: pointer;
  user-select: none;
  -webkit-user-select: none;
  outline: 0;
  transition: background-color .16s ease, border-color .16s ease, box-shadow .16s ease, color .16s ease;
  box-shadow: 0 0 0 0 rgba(138,180,248,0);
}

.fc-chip.is-active {
  background: #e8f0ff;
  border-color: #8ab4f8;
  color: #103a7a;
  font-weight: 600;
  box-shadow: 0 0 0 2px rgba(138,180,248,.35);
}

.fc-chip.util { opacity: 0.9 }

.fc-chart {
  width: 100%;
  min-height: 380px;
  min-width: 0;
  display: block;
  background: #fff;
  border-radius: 6px;
  box-shadow: inset 0 0 0 1px rgba(0,0,0,0.03);
}

.fc-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #7a869a;
  pointer-events: none;
  font-size: 13px;
  letter-spacing: .2px;
}
</style>
