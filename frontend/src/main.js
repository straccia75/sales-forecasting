// src/main.js
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
// keep if you already have a router; else remove these two lines
import router from './router'
import './assets/tailwind.css'

// mount
createApp(App).use(createPinia()).use(router).mount('#app')
