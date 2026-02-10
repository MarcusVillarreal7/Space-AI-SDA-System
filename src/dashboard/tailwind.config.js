/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        tier: {
          minimal: '#22c55e',
          low: '#84cc16',
          moderate: '#eab308',
          elevated: '#f97316',
          critical: '#ef4444',
        },
        space: {
          900: '#0a0e1a',
          800: '#111827',
          700: '#1e293b',
          600: '#334155',
          500: '#475569',
        },
      },
    },
  },
  plugins: [],
};
