/**
 * Tab switching between Charts, Summary, Events.
 */

import { state } from './state.js';
import { loadEvents } from './events.js';
import { loadSummary } from './summary.js';

export function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`tab-${name}`).classList.add('active');
    document.getElementById(`panel-${name}`).classList.add('active');
    if (name === 'events' && state.selectedRuns.length) loadEvents();
    if (name === 'summary' && state.selectedRuns.length) loadSummary();
}
