import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: { port: 5173 },
  optimizeDeps: {
    include: ['chart.js', 'chartjs-adapter-date-fns', 'date-fns'],
  },
})
