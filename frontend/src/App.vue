<!-- src/App.vue -->
<template>
  <div class="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 text-slate-900">
    <!-- Top navbar -->
    <header class="sticky top-0 z-40 border-b bg-white/80 backdrop-blur">
      <nav class="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
        <div class="flex items-center gap-2">
          <div class="h-6 w-6 rounded-lg bg-slate-900"></div>
          <span class="text-sm font-semibold tracking-tight">Sales Forecasting AI</span>
        </div>
        <div class="flex items-center gap-4 text-sm">
          <a href="#" class="text-slate-600 hover:text-slate-900">Docs</a>
          <a href="#" class="text-slate-600 hover:text-slate-900">Demo Data</a>
          <a href="#" class="text-slate-600 hover:text-slate-900">GitHub</a>
        </div>
      </nav>
    </header>

    <!-- Value prop -->
    <section class="mx-auto max-w-7xl px-4 pt-8 pb-4">
      <h1 class="text-2xl font-bold tracking-tight">Forecast demand in minutes, not weeks.</h1>
      <p class="mt-1 text-sm text-slate-600">
        Upload a CSV, map your columns, and generate interactive forecasts with confidence bands.
      </p>
    </section>

    <!-- Workspace: stacked flow; uploader collapses after forecast -->
    <main class="mx-auto max-w-7xl px-4 pb-6 space-y-4">
      <!-- Collapsible uploader shell -->
      <section class="rounded-2xl border bg-white shadow-sm overflow-hidden">
        <!-- Compact header that changes by stage -->
        <div class="flex items-center justify-between px-4 py-3 border-b">
          <div class="flex items-center gap-2">
            <span class="inline-flex h-2.5 w-2.5 rounded-full"
                  :class="stage==='results' ? 'bg-emerald-500' : 'bg-slate-400'"></span>
            <span class="text-sm font-medium">
              {{ stage==='results' ? 'Forecast ready' : 'Upload your data' }}
            </span>
            <span v-if="stage==='results' && resp?.used_model" class="ml-2 text-xs text-slate-500">
              (Model: {{ resp.used_model }})
            </span>
          </div>
          <div class="flex items-center gap-2">
            <button
              v-if="stage==='results'"
              class="rounded-md border px-2 py-1 text-xs hover:bg-slate-50"
              @click="expandUploader = !expandUploader"
            >
              {{ expandUploader ? 'Hide parameters' : 'Adjust parameters' }}
            </button>
            <button
              v-if="stage==='results'"
              class="rounded-md border px-2 py-1 text-xs hover:bg-slate-50"
              @click="resetToUpload"
            >
              Upload another
            </button>
          </div>
        </div>

        <!-- Uploader body: visible initially; collapsible when results exist -->
        <Transition name="fade-vert">
          <div v-show="stage==='upload' || expandUploader" class="p-4">
            <!-- CsvUploader emits { response } on Forecast -->
            <CsvUploader @ready="onReady" />
          </div>
        </Transition>
      </section>

      <!-- Results: rendered ONLY after forecast -->
      <Transition name="fade-vert">
        <section v-if="stage==='results'" class="rounded-2xl border bg-white p-4 shadow-sm">
          <!-- Badges -->
          <div class="mb-3 flex flex-wrap items-center gap-2">
            <span v-if="resp?.used_model" class="inline-flex items-center rounded-full border px-2 py-0.5 text-xs">
              Model: <strong class="ml-1">{{ resp.used_model }}</strong>
            </span>
            <span v-if="resp?.detected_frequency" class="inline-flex items-center rounded-full border px-2 py-0.5 text-xs">
              Freq: <strong class="ml-1">{{ resp.detected_frequency }}</strong>
            </span>
          </div>

          <!-- Metrics -->
          <div v-if="metricsAvailable" class="mb-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div class="rounded-xl border p-3">
              <div class="text-xs text-slate-500">MAPE</div>
              <div class="text-xl font-semibold">{{ fmt(metrics.mape) }}</div>
            </div>
            <div class="rounded-xl border p-3">
              <div class="text-xs text-slate-500">MAE</div>
              <div class="text-xl font-semibold">{{ fmt(metrics.mae) }}</div>
            </div>
            <div class="rounded-xl border p-3">
              <div class="text-xs text-slate-500">sMAPE</div>
              <div class="text-xl font-semibold">{{ fmt(metrics.smape) }}</div>
            </div>
          </div>

          <!-- Post-run store filter chips -->
          <div v-if="groups.length" class="mb-3 flex flex-wrap items-center gap-2">
            <div class="text-xs font-medium text-slate-600">Stores:</div>
            <button
              v-for="g in groups" :key="labelToKey[g]"
              @click="toggleGroup(labelToKey[g])"
              :class="[
                'rounded-full border px-2 py-1 text-xs transition',
                selectedGroups.has(labelToKey[g]) ? 'bg-slate-900 text-white border-slate-900' : 'hover:bg-slate-100'
              ]"  
              :aria-pressed="selectedGroups.has(labelToKey[g])"
            >
              {{ g }}
            </button>
            <div class="ml-auto flex gap-2">
              <button class="rounded-md border px-2 py-1 text-xs hover:bg-slate-50" @click="selectAll">All</button>
              <button class="rounded-md border px-2 py-1 text-xs hover:bg-slate-50" @click="selectNone">None</button>
            </div>
          </div>

          <!-- Chart -->
          <div class="overflow-hidden rounded-xl border">
            <ForecastChart
              v-if="resp"
              :response="resp"
              v-model:activeGroups="activeGroups"
            />
          </div>

          <!-- Warnings -->
          <div v-if="resp?.warnings?.length" class="mt-3 rounded-xl border border-amber-300 bg-amber-50 p-3 text-amber-800">
            <div class="mb-1 text-sm font-medium">Warnings</div>
            <ul class="list-disc pl-5 text-sm">
              <li v-for="(w,i) in resp.warnings" :key="i">{{ w }}</li>
            </ul>
          </div>
        </section>
      </Transition>
    </main>
  </div>
</template>


<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import CsvUploader from './components/CsvUploader.vue'
import ForecastChart from './components/ForecastChart.vue'

/* ---------------- Types ---------------- */
type ForecastPoint = {
  date: string
  yhat: number
  yhat_lower?: number|null
  yhat_upper?: number|null
  kind: 'history'|'forecast'
  group?: Record<string, any> | null
}
type ForecastResponse = {
  used_model: string
  detected_frequency: 'D'|'W'|'M'
  forecast: ForecastPoint[]
  metrics_overall?: { mape?: number|null, mae?: number|null, smape?: number|null }
  warnings?: string[]
}

/* ---------------- Data ---------------- */
const resp = ref<ForecastResponse|null>(null)
const metrics = computed(() => resp.value?.metrics_overall || ({ mape: null, mae: null, smape: null }))
const metricsAvailable = computed(() => !!resp.value?.metrics_overall)

/* ---------------- Group keys + selection ---------------- */
// ★ Use a canonical key for any string to avoid tiny mismatches
const keyOf = (s: string) => s.trim().toLowerCase()

// Groups are stored as display labels (strings), but we normalize them to keys when selecting
const groups = ref<string[]>([])
const selectedGroups = ref<Set<string>>(new Set()) // stores KEYS

/* Bridge for ForecastChart v-model:active-groups (array ↔ Set) */
const activeGroups = computed<string[]>({
  // ★ Always expose KEYS (not labels) to the child chart
  get: () => Array.from(selectedGroups.value),
  set: (val) => { selectedGroups.value = new Set((val || []).map(keyOf)) }
})

/* ---------------- Helpers ---------------- */
function keyOfGroup (g: Record<string, any> | null | undefined): string {
  if (!g || typeof g !== 'object') return 'All'
  // Keep your original label format, but selection will use a normalized key
  return Object.entries(g).map(([k,v]) => `${k}:${String(v)}`).join('|') || 'All'
}

// ★ Map from label -> key (stable, lowercase, trimmed)
const labelToKey = computed<Record<string, string>>(() =>
  Object.fromEntries(groups.value.map(label => [label, keyOf(label)]))
)

/* Recompute groups from response and seed selection safely */
function recomputeGroups() {
  const labels = new Set<string>()
  resp.value?.forecast?.forEach(p => labels.add(keyOfGroup(p.group)))
  groups.value = [...labels]                  // preserve your original no-sort behavior

  // ★ Seed selection to "all" using KEYS
  selectedGroups.value = new Set(groups.value.map(keyOf))
}

/* This gets called by CsvUploader when Forecast completes */
function onReady (e: { response: ForecastResponse }) {
  resp.value = e.response                     // <-- unchanged (preserves drawing behavior)
  stage.value = 'results'                     // show results panel
  expandUploader.value = false                // collapse uploader
}
watch(resp, recomputeGroups)

/* ★ Keep selection consistent if the groups list changes later (e.g., new data) */
watch(groups, (gs) => {
  const allowed = new Set(gs.map(keyOf))
  // filter current selection to allowed keys
  const next = new Set([...selectedGroups.value].filter(k => allowed.has(k)))
  // if nothing remains selected but there are groups, default to "all"
  if (next.size === 0 && gs.length) for (const label of gs) next.add(keyOf(label))
  selectedGroups.value = next
})

/* ---------------- Chip controls ---------------- */
// ★ Operate on KEYS (callers should pass keys)
function toggleGroup(g: string){
  const k = keyOf(g)
  const s = new Set(selectedGroups.value)
  s.has(k) ? s.delete(k) : s.add(k)
  selectedGroups.value = s
}
function selectAll(){ selectedGroups.value = new Set(groups.value.map(keyOf)) }
function selectNone(){ selectedGroups.value = new Set() }
function fmt(v: number|null|undefined){
  if(v==null) return '—'
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: 3 })
}

/* ---------------- View state for new template ---------------- */
const stage = ref<'upload' | 'results'>('upload')
const expandUploader = ref(false)

function resetToUpload () {
  resp.value = null
  groups.value = []
  selectedGroups.value = new Set()
  stage.value = 'upload'
  expandUploader.value = true
}
</script>


