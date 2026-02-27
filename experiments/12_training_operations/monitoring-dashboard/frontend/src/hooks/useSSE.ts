import { useEffect, useRef } from 'react';
import { useMetricsStore } from '../stores/metricsStore';

const MAX_RETRY_DELAY = 30_000;

export function useSSE() {
  const applySnapshot = useMetricsStore((s) => s.applySnapshot);
  const applyDelta = useMetricsStore((s) => s.applyDelta);
  const applyRunsMeta = useMetricsStore((s) => s.applyRunsMeta);
  const setConnectionStatus = useMetricsStore((s) => s.setConnectionStatus);
  const retryDelay = useRef(1000);

  useEffect(() => {
    let es: EventSource | null = null;
    let retryTimeout: ReturnType<typeof setTimeout> | null = null;
    let disposed = false;

    function connect() {
      if (disposed) return;
      setConnectionStatus('connecting');
      es = new EventSource('/stream');

      es.addEventListener('snapshot', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          applySnapshot(data);
          setConnectionStatus('connected');
          retryDelay.current = 1000;
        } catch (err) {
          console.error('Failed to parse snapshot:', err);
        }
      });

      es.addEventListener('delta', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          applyDelta(data);
        } catch (err) {
          console.error('Failed to parse delta:', err);
        }
      });

      es.addEventListener('runs_meta', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data);
          applyRunsMeta(data.runs_meta);
        } catch (err) {
          console.error('Failed to parse runs_meta:', err);
        }
      });

      es.onerror = () => {
        es?.close();
        setConnectionStatus('disconnected');
        if (!disposed) {
          retryTimeout = setTimeout(() => {
            retryDelay.current = Math.min(retryDelay.current * 2, MAX_RETRY_DELAY);
            connect();
          }, retryDelay.current);
        }
      };
    }

    connect();

    return () => {
      disposed = true;
      es?.close();
      if (retryTimeout) clearTimeout(retryTimeout);
    };
  }, [applySnapshot, applyDelta, applyRunsMeta, setConnectionStatus]);
}
