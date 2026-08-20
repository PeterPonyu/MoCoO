import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        paper: '#f3efe4',
        ink: '#1c1917',
        rust: '#9a3412',
        brand: '#9a3412',
      },
    },
  },
  plugins: [],
};

export default config;
