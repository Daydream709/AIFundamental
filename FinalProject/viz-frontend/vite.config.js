import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath, URL } from 'node:url';
export default defineConfig({
    plugins: [vue(), tailwindcss()],
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url)),
        },
    },
    server: {
        port: 5173,
        host: '0.0.0.0',
    },
    build: {
        // Split heavy vendor chunks so the main entry doesn't blow past 500KB.
        // BUGFIX: previously a single 932KB index-*.js block was shipped; on
        // cold start / slow networks this looked like "界面打不开" because
        // parse + initialize echarts + naive-ui took several seconds.
        rollupOptions: {
            output: {
                manualChunks: {
                    'vendor-vue': ['vue', 'vue-router', 'pinia', '@vueuse/core'],
                    'vendor-echarts': ['echarts/core', 'echarts/charts', 'echarts/components', 'echarts/renderers', 'vue-echarts'],
                    'vendor-naive': ['naive-ui', '@formkit/auto-animate', '@vueuse/motion'],
                },
            },
        },
        chunkSizeWarningLimit: 600,
    },
});
