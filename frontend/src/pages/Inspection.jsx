import { useState, useEffect } from 'react'
import { startSystem, stopSystem, getStatus, getDetected, liveProcessUrl } from '../api/client'
import styles from './Inspection.module.css'

export default function Inspection() {
  const [running, setRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [detection, setDetection] = useState(null)
  const [streamKey, setStreamKey] = useState(0)
  const [error, setError] = useState(null)

  // Restore state when navigating back to this page
  useEffect(() => {
    getStatus().then(s => {
      if (s.running) {
        setRunning(true)
        setStreamKey(k => k + 1)
        getDetected().then(d => {
          if (d && d.final_status) setDetection(d)
        }).catch(() => {})
      }
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!running) return

    const id = setInterval(async () => {
      try {
        const data = await getDetected()
        if (data && data.final_status) {
          setDetection(data)
        }
      } catch {
        // backend unreachable — ignore, keep trying
      }
    }, 1000)

    return () => clearInterval(id)
  }, [running])

  const handleStart = async () => {
    if (loading) return
    setError(null)
    setLoading(true)
    try {
      await startSystem()
      setStreamKey(k => k + 1)
      setDetection(null)
      setRunning(true)
    } catch {
      setError('Could not connect to backend at localhost:8000')
    } finally {
      setLoading(false)
    }
  }

  const handleStop = async () => {
    if (loading) return
    setLoading(true)
    try {
      await stopSystem()
    } finally {
      setRunning(false)
      setDetection(null)
      setLoading(false)
    }
  }

  const status = detection?.final_status
  const isPass   = status === 'PASS'
  const isReject = status === 'REJECT'
  const isNoPart = status === 'NO_PART'

  return (
    <div className={styles.page}>
      {/* PAGE HEADER */}
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.pageTitle}>Inspection</h1>
          <p className={styles.pageDesc}>Hardware-triggered camera inspection</p>
        </div>
        <span className={running ? `${styles.badge} ${styles.badgeRunning}` : `${styles.badge} ${styles.badgeStopped}`}>
          <span className={styles.badgeDot} />
          {running ? 'Running' : 'Stopped'}
        </span>
      </div>

      {error && (
        <div className={styles.errorBar}>{error}</div>
      )}

      {/* MAIN BODY */}
      <div className={styles.body}>

        {/* VIDEO CARD */}
        <div className={styles.videoCard}>
          <div className={styles.cardHeader}>
            <span>Last Inspection Result</span>
            {detection && (
              <span className={isPass ? `${styles.resultPill} ${styles.pillPass}` : `${styles.resultPill} ${styles.pillReject}`}>
                {isPass ? 'PASS' : isReject ? 'REJECT' : status}
              </span>
            )}
          </div>
          <div className={styles.feedArea}>
            {running && detection ? (
              <img
                key={streamKey}
                src={liveProcessUrl()}
                className={styles.cameraFeed}
                alt="Last inspection result"
              />
            ) : running ? (
              <div className={styles.placeholder}>
                <div className={styles.placeholderIcon}>
                  <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                    <circle cx="12" cy="13" r="4" />
                  </svg>
                </div>
                <p className={styles.placeholderText}>Waiting for hardware trigger...</p>
                <p className={styles.placeholderSub}>Camera armed on {' '}
                  <code className={styles.code}>Line0 / RisingEdge</code>
                </p>
              </div>
            ) : (
              <div className={styles.placeholder}>
                <div className={styles.placeholderIcon}>
                  <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                    <circle cx="12" cy="13" r="4" />
                  </svg>
                </div>
                <p className={styles.placeholderText}>System stopped</p>
                <p className={styles.placeholderSub}>Press START to connect the camera</p>
              </div>
            )}
          </div>
        </div>

        {/* SIDE PANEL */}
        <div className={styles.sidePanel}>

          {/* CONTROLS */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>Controls</div>
            <div className={styles.btnGroup}>
              <button
                className={`${styles.btn} ${styles.btnStart}`}
                onClick={handleStart}
                disabled={running || loading}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5,3 19,12 5,21" />
                </svg>
                {loading && !running ? 'STARTING...' : 'START'}
              </button>
              <button
                className={`${styles.btn} ${styles.btnStop}`}
                onClick={handleStop}
                disabled={!running || loading}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                </svg>
                {loading && running ? 'STOPPING...' : 'STOP'}
              </button>
            </div>
          </div>

          {/* RESULT */}
          <div className={styles.card}>
            <div className={styles.cardTitle}>Result</div>

            {detection ? (
              <div className={styles.resultList}>
                <div className={styles.resultRow}>
                  <span className={styles.rowLabel}>Part</span>
                  <span className={styles.rowValue}>{detection.part_id ?? '—'}</span>
                </div>
                <div className={styles.resultRow}>
                  <span className={styles.rowLabel}>Status</span>
                  <span className={`${styles.rowValue} ${isPass ? styles.colorPass : isReject ? styles.colorReject : styles.colorWarn}`}>
                    {isPass ? 'PASS' : isReject ? 'REJECT' : isNoPart ? 'NO PART' : status}
                  </span>
                </div>
                <div className={styles.resultRow}>
                  <span className={styles.rowLabel}>QC</span>
                  <span className={`${styles.rowValue} ${detection.qc_fail ? styles.colorReject : styles.colorPass}`}>
                    {detection.qc_fail ? 'FAIL' : 'OK'}
                  </span>
                </div>
                <div className={styles.resultRow}>
                  <span className={styles.rowLabel}>Score</span>
                  <span className={styles.rowValue}>
                    {typeof detection.score === 'number'
                      ? `${(detection.score * 100).toFixed(1)}%`
                      : '—'}
                  </span>
                </div>
                <div className={styles.resultRow}>
                  <span className={styles.rowLabel}>Frame</span>
                  <span className={styles.rowValue}>
                    {detection.frame_count ?? '—'}
                  </span>
                </div>
                <div className={styles.resultRow}>
                  <span className={styles.rowLabel}>Workers</span>
                  <span className={styles.rowValue}>
                    <span className={styles.workerBadge}>
                      {detection.active_worker ?? '—'}
                      <span className={styles.workerSep}>/</span>
                      {detection.total_worker ?? '—'}
                    </span>
                  </span>
                </div>
              </div>
            ) : (
              <p className={styles.noResult}>
                {running ? 'Waiting for first trigger...' : 'No result yet'}
              </p>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}
