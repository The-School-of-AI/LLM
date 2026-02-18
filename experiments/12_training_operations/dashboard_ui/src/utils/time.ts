import { format, formatDistanceToNow, parseISO } from 'date-fns';

export function formatDuration(seconds: number): string {
    if (seconds < 0) return '00:00:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);

    const pad = (n: number) => n.toString().padStart(2, '0');
    return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

export function formatDateTime(isoString: string): string {
    try {
        return format(parseISO(isoString), 'MMM d, HH:mm:ss');
    } catch {
        return 'Invalid Date';
    }
}

export function getTimeAgo(isoString: string): string {
    try {
        return formatDistanceToNow(parseISO(isoString), { addSuffix: true });
    } catch {
        return 'Unknown time';
    }
}
