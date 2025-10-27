/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['index.html', 'src/**/*.{vue,js,ts}'],
  darkMode: ['class', '[data-theme="dark"]'], // choose how you toggle
  // If you often need your global utilities to beat SFC styles, uncomment:
  // important: true,
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)',
        elev: 'var(--elev)',
        text: 'var(--text)',
        muted: 'var(--muted)',
        primary: 'var(--primary)',
        border: 'var(--border)',
      },
      borderRadius: {
        DEFAULT: 'var(--radius)',
        lg: 'var(--radius-lg)',
      },
      boxShadow: {
        card: 'var(--shadow)',
      },
    },
  },
  safelist: [
    // if you build class names dynamically at runtime, safelist them
    'btn', 'btn-primary', 'card', 'app-surface'
  ],
  plugins: [],
}
