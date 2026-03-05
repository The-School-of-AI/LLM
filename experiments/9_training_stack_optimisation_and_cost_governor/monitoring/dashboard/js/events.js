/**
 * Events tab: load and render the event log.
 */

import { state } from './state.js';
import { fetchEvents } from './api.js';

export async function loadEvents() {
    const primary = state.selectedRuns[0];
    if (!primary) return;
    const el = document.getElementById('eventsContent');
    el.innerHTML = '<div class="empty"><div class="spinner"></div></div>';
    try {
        const evs = await fetchEvents(primary.run_id);
        if (!evs.length) {
            el.innerHTML = '<div class="empty"><div class="empty-icon">&#9672;</div><div class="empty-title">No events</div><div class="empty-sub">No events logged for this run</div></div>';
            return;
        }
        el.innerHTML = `
            <div class="events-header"><span>Time</span><span>Step</span><span>Type</span><span>Severity</span><span>Message</span></div>
            <div class="events-list">${evs.map(e => {
                const escapedMsg = escapeHtml(e.message || '');
                return `<div class="event-row">
                    <span class="event-time">${escapeHtml(e.time)}</span>
                    <span class="event-step">${Number(e.step).toLocaleString()}</span>
                    <span class="event-type">${escapeHtml(e.event_type)}</span>
                    <span class="sev-${e.severity || 'info'}">${escapeHtml(e.severity || 'info')}</span>
                    <span class="event-msg" title="${escapedMsg}">${escapedMsg}</span>
                </div>`;
            }).join('')}
            </div>`;
    } catch (e) {
        el.innerHTML = '<div class="empty"><span style="font-size:0.75rem;color:var(--muted)">Failed to load events</span></div>';
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
