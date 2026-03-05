/**
 * Pure utility functions — no side effects, no DOM access, no state mutation.
 */

import { RUN_COLORS } from './constants.js';
import { state } from './state.js';

/** Format a numeric value for display. */
export function fmt(v) {
    if (v == null) return '\u2014';
    return Math.abs(v) >= 1000 ? v.toFixed(1)
         : Math.abs(v) >= 1   ? v.toFixed(4)
         : v.toExponential(3);
}

/** Compute min/max/mean/latest/count from a data array. */
export function stats(data) {
    if (!data.length) return null;
    const isHist = Array.isArray(data[0]?.value);
    if (isHist) {
        const last = data[data.length - 1]?.value || [];
        if (!last.length) return null;
        const min = Math.min(...last), max = Math.max(...last);
        return { min, max, mean: last.reduce((a, b) => a + b, 0) / last.length, latest: max, count: data.length };
    }
    const v = data.map(d => d.y ?? d.value ?? 0);
    const min = Math.min(...v), max = Math.max(...v);
    return { min, max, mean: v.reduce((a, b) => a + b, 0) / v.length, latest: v[v.length - 1], count: v.length };
}

/** Get the color for a run by its stable index (position-based, for things like pie slices). */
export function getRunColor(runIdx) {
    return RUN_COLORS[runIdx % RUN_COLORS.length];
}

/** Get the stable color for a run by its run_id. Uses runColorMap in state. */
export function getRunColorById(runId) {
    const idx = state.runColorMap.get(runId) ?? 0;
    return RUN_COLORS[idx % RUN_COLORS.length];
}

/** Map a 0-1 normalized value to a cool-blue → hot-red color. */
export function heatColor(t) {
    if (t < 0.5) {
        const r = Math.round(13 + t * 2 * 40);
        const g = Math.round(17 + t * 2 * 60);
        const b = Math.round(80 + t * 2 * 100);
        return `rgb(${r},${g},${b})`;
    } else {
        const n = (t - 0.5) * 2;
        const r = Math.round(53 + n * 200);
        const g = Math.round(77 - n * 50);
        const b = Math.round(180 - n * 140);
        return `rgb(${r},${g},${b})`;
    }
}

/** Show a temporary toast notification. */
export function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2200);
}
