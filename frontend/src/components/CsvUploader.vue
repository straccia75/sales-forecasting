<!-- src/components/CsvUploader.vue -->
<template>
  <!--
    PURPOSE
    - Upload CSV, map columns, optionally select a single store to run,
      tune forecast parameters, and submit to FastAPI `/forecast`.

    KEY UX SECTIONS
    - Drag & drop zone
    - Mapping: date, target, group_by (store), regressors
    - Optional pre-run single store dropdown (only appears when group_by chosen)
    - Params: model, horizon, confidence, frequency, exog policy
    - Warnings pane (shows backend validation messages)
  -->
  <div class="space-y-5">
    <!-- ========================= DRAG & DROP ========================= -->
    <div
      class="group relative rounded-2xl border border-dashed p-6 text-center transition hover:border-slate-400"
      :class="rows.length ? 'border-slate-300 bg-slate-50/60' : 'border-slate-300 bg-white'"
      @dragover.prevent
      @dragenter.prevent
      @drop.prevent="onDrop"
    >
      <!-- Icon + microcopy -->
      <div class="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl border bg-white shadow-sm">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="h-6 w-6 text-slate-600">
          <path fill="currentColor" d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zm1 7h5l-6-6z"/>
        </svg>
      </div>
      <div class="text-sm">
        <!-- Hidden input gives native file picker -->
        <label class="cursor-pointer font-medium text-slate-900 underline-offset-2 hover:underline">
          <input class="sr-only" type="file" accept=".csv" @change="onFile" />
          Click to upload
        </label>
        <span class="mx-1 text-slate-500">or</span>
        <span class="text-slate-500">drag & drop your CSV</span>
      </div>
      <div v-if="fileName" class="mt-2 text-xs text-slate-500">{{ fileName }}</div>
      <button v-if="rows.length" @click="reset" class="absolute right-3 top-3 rounded-md border px-2 py-1 text-xs hover:bg-slate-50">
        Reset
      </button>
    
    <!-- ========================= DEMO DATASETS ========================= -->
    <div class="rounded-xl border p-4">
      <div class="mb-1 text-xs font-medium text-slate-500">Use a demo dataset</div>
      <div class="grid grid-cols-1 gap-2 md:grid-cols-3">
        <select v-model="selectedDemoId" class="w-full rounded-lg border bg-white p-2 text-sm">
          <option disabled value="">Select a demo dataset</option>
          <option v-for="d in demoOptions" :key="d.id" :value="d.id">{{ d.name }}</option>
        </select>
        <button
          @click="loadDemo"
          :disabled="!selectedDemoId || loadingDemo"
          class="rounded-lg border px-3 py-2 text-sm hover:bg-slate-50"
        >
          {{ loadingDemo ? 'Loading…' : 'Load demo' }}
        </button>
        <div class="text-xs text-slate-500 self-center" v-if="currentDemo && headers.length">
          Loaded: <strong>{{ fileName }}</strong>
        </div>
        <div v-if="demoError" class="text-xs text-red-600 md:col-span-3">{{ demoError }}</div>
      </div>
    </div>

</div>

    <!-- ========================= MAPPING ========================= -->
    <div v-if="headers.length" class="grid gap-4 md:grid-cols-2">
      <!-- REQUIRED: Date column -->
      <div class="rounded-xl border p-4">
        <div class="mb-1 text-xs font-medium text-slate-500">Date column</div>
        <select v-model="map.date" class="w-full rounded-lg border bg-white p-2 text-sm">
          <option disabled value="">Select date column</option>
          <option v-for="h in headers" :key="'d'+h" :value="h">{{ h }}</option>
        </select>
        <!-- Tip: YYYY-MM-DD strings are fine; backend validates. -->
      </div>

      <!-- REQUIRED: Target (numeric) -->
      <div class="rounded-xl border p-4">
        <div class="mb-1 text-xs font-medium text-slate-500">Target (y)</div>
        <select v-model="map.target" class="w-full rounded-lg border bg-white p-2 text-sm">
          <option disabled value="">Select target column</option>
          <option v-for="h in headers" :key="'t'+h" :value="h">{{ h }}</option>
        </select>
        <!-- Tip: I cast this to Number() client-side before POST. -->
      </div>

      <!-- OPTIONAL: Group-by (store) — UI holds one column name, backend expects a LIST -->
      <div class="rounded-xl border p-4">
        <div class="mb-1 text-xs font-medium text-slate-500">Group by (stores)</div>
        <select v-model="map.group_by" class="w-full rounded-lg border bg-white p-2 text-sm">
          <option value="">(Optional) choose a store column</option>
          <option v-for="h in headers" :key="'g'+h" :value="h">{{ h }}</option>
        </select>
        <!-- NOTE: On submit I wrap as [group_by] to match backend schema. -->
      </div>

      <!-- OPTIONAL: Regressors (multi-select) -->
      <div class="rounded-xl border p-4">
        <div class="mb-1 text-xs font-medium text-slate-500">Regressors</div>
        <select v-model="map.regressors" multiple class="w-full rounded-lg border bg-white p-2 text-sm min-h-[44px]">
          <option v-for="h in headers" :key="'r'+h" :value="h">{{ h }}</option>
        </select>
        <!-- Tweak: add/remove columns; backend ignores empty list. -->
      </div>
    </div>

    <!-- ===== PRE-RUN: single store (visible only if group_by chosen) ===== -->
    <div v-if="headers.length && map.group_by" class="rounded-xl border p-4">
      <div class="mb-1 text-xs font-medium text-slate-500">Run only this store (optional)</div>
      <select v-model="runStore" class="w-full rounded-lg border bg-white p-2 text-sm">
        <option value="">All stores</option>
        <option v-for="val in uniqueStoreValues" :key="val" :value="val">{{ val }}</option>
      </select>
      <!-- If set, I filter rows client-side so backend trains only on that store. -->
    </div>

    <!-- ========================= FORECAST PARAMS ========================= -->
    <div v-if="headers.length" class="grid gap-4 md:grid-cols-3">
      <!-- Model family -->
      <div class="rounded-xl border p-4">
        <div class="mb-1 text-xs font-medium text-slate-500">Model</div>
        <select v-model="params.model_preference" class="w-full rounded-lg border bg-white p-2 text-sm">
          <option value="auto">auto</option>
          <option value="prophet">prophet</option>
          <option value="ets">ets</option>
          <option value="sarimax">sarimax</option>
          <option value="seasonal-naive">seasonal-naive</option>
          <option value="auto-ets-sarimax">auto-ets-sarimax</option>
        </select>
        <p v-if="params.model_preference==='ets'" class="mt-1 text-xs text-slate-500">ETS ignores regressors.</p>
      </div>

      <!-- Forecast horizon -->
      <div class="rounded-xl border p-4">
        <div class="mb-1 text-xs font-medium text-slate-500">Horizon</div>
        <input type="number" min="1" v-model.number="params.horizon" class="w-full rounded-lg border bg-white p-2 text-sm" />
        <!-- Tweak: higher = longer forecast window. -->
      </div>

      <!-- Confidence interval width -->
      <div class="rounded-xl border p-4">
        <div class="mb-1 text-xs font-medium text-slate-500">Confidence (0..1)</div>
        <input type="number" step="0.05" min="0.5" max="0.99" v-model.number="params.confidence" class="w-full rounded-lg border bg-white p-2 text-sm" />
        <!-- Tweak: closer to 1 = wider uncertainty band. -->
      </div>

      <!-- Frequency override (usually leave auto) -->
      <div class="rounded-xl border p-4">
        <div class="mb-1 text-xs font-medium text-slate-500">Frequency</div>
        <select v-model="params.frequency" class="w-full rounded-lg border bg-white p-2 text-sm">
          <option value="auto">auto</option>
          <option value="D">Daily</option>
          <option value="W">Weekly</option>
          <option value="M">Monthly</option>
        </select>
      </div>

      <!-- How to fill unknown future regressor values -->
      <div class="rounded-xl border p-4">
        <div class="mb-1 text-xs font-medium text-slate-500">Future exog policy</div>
        <select v-model="params.exog_future_policy" class="w-full rounded-lg border bg-white p-2 text-sm">
          <option value="zero">zero</option>
          <option value="last">last</option>
          <option value="mean">mean</option>
        </select>
      </div>
    </div>

    <!-- ========================= SUBMIT ========================= -->
<div v-if="headers.length" class="flex items-center gap-3">
  <button
    :disabled="!canSubmit || busy"
    @click="submit"
    :class="[
      'rounded-lg px-4 py-2 text-sm font-medium text-white shadow-sm transition',
      busy ? 'bg-slate-400' : 'bg-slate-900 hover:bg-slate-800'
    ]"
  >
    {{ busy ? 'Running…' : 'Forecast' }}
  </button>

  <!-- Determinate (0–100) -->
  <div
    v-if="busy && typeof progress === 'number'"
    class="progress-wrap"
    :title="`${progress}%`"
  >
    <div
      class="progress"
      role="progressbar"
      :aria-valuemin="0"
      :aria-valuemax="100"
      :aria-valuenow="progress"
      :style="{ '--w': progress + '%' }"
    />
    <span class="progress-text">{{ progress }}%</span>
  </div>

  <!-- Indeterminate -->
  <div v-else-if="busy" class="progress-wrap" title="Working…">
    <div class="progress indeterminate" role="progressbar" aria-busy="true" />
    <span class="progress-text">Working…</span>
  </div>

  <span v-if="responseWarnings.length" class="text-xs text-amber-700">
    Check warnings below
  </span>
</div>


    <!-- ========================= WARNINGS ========================= -->
    <div v-if="responseWarnings.length" class="rounded-xl border border-amber-300 bg-amber-50 p-3 text-amber-800">
      <div class="mb-1 text-sm font-medium">Warnings</div>
      <ul class="list-disc pl-5 text-sm">
        <li v-for="(w,i) in responseWarnings" :key="i">{{ w }}</li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
import Papa from 'papaparse'
import { ref, computed } from 'vue'

/** TYPES (lightweight; mirror backend at a high level) */
type Row = Record<string, any>
type ForecastPoint = {
  date:string; yhat:number; yhat_lower?:number|null; yhat_upper?:number|null;
  kind:'history'|'forecast'; group?:Record<string, any>|null
}
type ForecastResponse = {
  used_model:string; detected_frequency:'D'|'W'|'M'; forecast:ForecastPoint[];
  metrics_overall?:{ mape?:number|null, mae?:number|null, smape?:number|null };
  warnings?:string[]
}

/** PROPS & EMITS
 *  - apiBase (optional): override backend base; else read VITE_API_BASE from .env.local
 *  - emits: 'ready' with the entire ForecastResponse so parent can render chart
 */
const props = defineProps<{ apiBase?: string; demoDatasets?: { id:string; name:string; url:string }[] }>()
const emit  = defineEmits<{ (e:'ready', payload:{ response: ForecastResponse }): void }>()

/** API base resolution order: prop > .env.local (VITE_API_BASE) > '' (same-origin) */
const apiBaseResolved = computed(() => props.apiBase ?? (import.meta as any).env?.VITE_API_BASE ?? '')

/** STATE */
const rows     = ref<Row[]>([])
const headers  = ref<string[]>([])
const fileName = ref<string>('')

/** Column mapping
 *  - group_by in UI is a single string (column name), but backend expects a LIST → wrap on submit.
 */
const map = ref<{ date:string; target:string; group_by:string|''; regressors:string[] }>({
  date:'', target:'', group_by:'', regressors:[]
})

/** Optional: pre-run single store */
const runStore = ref<string>('')

/** Parameters (user-tunable) */
const params = ref<{
  model_preference:string; horizon:number; confidence:number;
  frequency:'auto'|'D'|'W'|'M'; exog_future_policy:'zero'|'last'|'mean'
}>({
  model_preference:'auto', // 'auto' | 'prophet' | 'ets' | 'sarimax' | 'seasonal-naive' | 'auto-ets-sarimax'
  horizon:30,              // future periods (# of D/W/M depending on frequency)
  confidence:0.8,          // 0.5..0.99 -> band width
  frequency:'auto',        // override if your CSV cadence is known
  exog_future_policy:'last'// 'zero' | 'last' | 'mean' for future regressors
})

const responseWarnings = ref<string[]>([])
const busy  = ref(false)
const canSubmit = computed(() => !!rows.value.length && map.value.date && map.value.target)


/** DEMO DATASETS (optional, works like a virtual upload) */
type DemoItem = { id:string; name:string; url:string }
const defaultDemoOptions = ref<DemoItem[]>([
  { id:'walmart-retail-sales', name:'Walmart Sales', url:'/demo/walmart_sales.csv' },
  { id:'retail-sales-multi-store', name:'WOmart Retail Sales', url:'/demo/WOmart_Data_Sales.csv' },
  { id:'retail-forecasting', name:'Sales Dataset Superstore', url:'/demo/Sales_dataset.csv' }
])
const demoOptions = computed<DemoItem[]>(() => (props.demoDatasets?.length ? props.demoDatasets! : defaultDemoOptions.value))
const selectedDemoId = ref<string>('')
const loadingDemo = ref(false)
const demoError = ref<string>('')
const currentDemo = computed<DemoItem | null>(() => demoOptions.value.find(d => d.id === selectedDemoId.value) || null)

async function loadDemo(){
  demoError.value = ''
  if(!currentDemo.value) return
  try{
    loadingDemo.value = true
    const res = await fetch(currentDemo.value.url, { cache:'no-store' })
    if(!res.ok) throw new Error(`Failed to load demo CSV (${res.status})`)
    const text = await res.text()
    const parsed = Papa.parse<Row>(text, { header:true, skipEmptyLines:true })
    rows.value = (parsed.data as Row[]) || []
    headers.value = parsed.meta.fields || []
    fileName.value = `${currentDemo.value.name}.csv`
    if (map.value.group_by && !headers.value.includes(map.value.group_by)) map.value.group_by = ''
    runStore.value = ''
  }catch(err:any){
    demoError.value = String(err?.message || err) || 'Could not load demo dataset.'
  }finally{
    loadingDemo.value = false
  }
}
/** FILE HANDLERS */
function onDrop(e:DragEvent){ const f = e.dataTransfer?.files?.[0]; if(f) readFile(f) }
function onFile(e:any){     const f = e.target?.files?.[0];          if(f) readFile(f) }

function readFile(file:File){
  fileName.value = file.name
  Papa.parse<Row>(file, {
    header:true, skipEmptyLines:true,
    complete: (res)=>{
      rows.value    = res.data
      headers.value = res.meta.fields || []
      // If previous group_by doesn’t exist in new file, reset it
      if (map.value.group_by && !headers.value.includes(map.value.group_by)) map.value.group_by = ''
      runStore.value = ''
    }
  })
}

/** RESET current dataset + mapping */
function reset(){
  rows.value = []; headers.value = []; fileName.value = ''
  map.value  = { date:'', target:'', group_by:'', regressors:[] }
  runStore.value = ''; responseWarnings.value = []
}

const isRunning = ref(false)
// Use a number 0–100 for determinate mode, or set to null for indeterminate
const progress = ref<number | null>(null)

async function run() {
  isRunning.value = true
  // EXAMPLE 1: known steps -> determinate
  // const steps = 10
  // for (let i = 0; i < steps; i++) {
  //   await doStep(i)
  //   progress.value = Math.round(((i + 1) / steps) * 100)
  // }
  // isRunning.value = false

  // EXAMPLE 2: unknown length -> indeterminate
  progress.value = null
  await doSomethingLong()
  isRunning.value = false
}

// Hook these if applicable:
// worker.onmessage = (e) => {
//   if (e.data.type === 'progress') progress.value = e.data.value // 0–100
//   if (e.data.type === 'done') { isRunning.value = false }
// }
// xhr.upload.onprogress = (e) => {
//   if (e.lengthComputable) progress.value = Math.round((e.loaded / e.total) * 100)
// }

/** SUBMIT to FastAPI `/forecast` */
async function submit(){
  try{
    busy.value = true
    responseWarnings.value = []

    // (1) Optional: single-store pre-filter (so backend trains only that subset)
    const groupCol  = map.value.group_by || ''
    let rowsToSend  = rows.value
    if (groupCol && runStore.value) {
      rowsToSend = rows.value.filter(r => String(r[groupCol]) === runStore.value)
    }

    // (2) Normalize numeric target (avoid strings slipping through)
    const normalized = rowsToSend.map(r => {
      const copy:any = { ...r }
      if (copy[map.value.target] != null) copy[map.value.target] = Number(copy[map.value.target])
      return copy
    })

    // (3) Build payload EXACTLY as backend expects:
    //     - data: list of rows
    //     - schema.group_by: LIST (or null)
    const payload = {
      data: normalized,
      schema: {
        date: map.value.date,
        target: map.value.target,
        group_by: groupCol ? [groupCol] : null,       // <-- LIST here (fixes your 422)
        regressors: (map.value.regressors || []).filter(Boolean)
      },
      params: {
        model_preference: params.value.model_preference.toLowerCase(),
        horizon:    params.value.horizon,
        confidence: params.value.confidence,
        frequency:  params.value.frequency,
        exog_future_policy: params.value.exog_future_policy
      }
    }

    // (4) POST to backend
    const res = await fetch(`${apiBaseResolved.value}/forecast`, {
      method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify(payload)
    })
    if(!res.ok){ throw new Error(await res.text() || `HTTP ${res.status}`) }

    // (5) Emit to parent & surface warnings
    const json = await res.json() as ForecastResponse
    responseWarnings.value = json.warnings || []
    emit('ready', { response: json })
  }catch(err:any){
    responseWarnings.value = [String(err?.message || err)]
  } finally {
    busy.value = false
  }
}

/** Store values for the pre-run dropdown */
const uniqueStoreValues = computed(() => {
  const col = map.value.group_by
  if (!col) return [] as string[]
  const s = new Set<string>()
  for (const r of rows.value) if (r[col] != null) s.add(String(r[col]))
  return [...s]
})
</script>
