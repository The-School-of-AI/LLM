import { useState, useRef, useEffect } from 'react'

export function InfoBadge({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  return (
    <div className="info-badge-wrap" ref={ref}>
      <button
        className="info-badge-btn"
        onClick={() => setOpen(v => !v)}
        aria-label="Info"
      >
        i
      </button>
      {open && (
        <div className="info-badge-tip">
          {text}
        </div>
      )}
    </div>
  )
}
