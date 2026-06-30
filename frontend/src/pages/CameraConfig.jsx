import { useState, useEffect } from 'react'
import { getConfig, saveConfig } from '../api/client'
import styles from './CameraConfig.module.css'

export default function CameraConfig() {
  const [form,   setForm]   = useState(null)
  const [error,  setError]  = useState(null)
  const [dirty,  setDirty]  = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved,  setSaved]  = useState(false)

  useEffect(() => {
    getConfig()
      .then(data => setForm(data))
      .catch(() => setError('Could not load config from backend.'))
  }, [])

  const set = (key, value) => {
    setForm(prev => ({ ...prev, [key]: value }))
    setDirty(true)
    setSaved(false)
  }

  const handleSave = async () => {
    setSaving(true)
    setError(null)
    try {
      await saveConfig(form)
      setSaved(true)
      setDirty(false)
    } catch {
      setError('Failed to save. Is the backend running?')
    } finally {
      setSaving(false)
    }
  }

  if (!form && !error) {
    return (
      <div className={styles.page}>
        <div className={styles.pageHeader}>
          <h1 className={styles.pageTitle}>Camera Config</h1>
        </div>
        <div className={styles.skeletonGrid}>
          {[200, 340, 220, 300, 200, 200].map((h, i) => (
            <div key={i} className={styles.skeleton} style={{ height: h }} />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className={styles.page}>

      {/* ── Header ── */}
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.pageTitle}>Camera Config</h1>
          <p className={styles.pageDesc}>Edit and save inspection settings</p>
        </div>
        <button
          className={`${styles.saveBtn} ${saved ? styles.saveBtnSaved : ''}`}
          onClick={handleSave}
          disabled={!dirty || saving}
        >
          {saving ? (
            <><span className={styles.spinner} /> Saving…</>
          ) : saved ? (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              Saved
            </>
          ) : 'Save Changes'}
        </button>
      </div>

      {error && <div className={styles.errorBar}>{error}</div>}

      {/* ── Grid of section cards ── */}
      <div className={styles.pageGrid}>

        {/* ── Trigger ── */}
        <Card title="Trigger" color="#f59e0b">
          <Toggle label="External Trigger" value={form.use_external_trigger} onChange={v => set('use_external_trigger', v)} />
          <Text   label="Trigger Source"   value={form.trigger_source}       onChange={v => set('trigger_source', v)}       hint="e.g. Line0" />
          <Text   label="Activation"       value={form.trigger_activation}   onChange={v => set('trigger_activation', v)}   hint="e.g. RisingEdge" />
        </Card>

        {/* ── Camera ── */}
        <Card title="Camera" color="#3b82f6">
          <Text   label="Device User ID"  value={form.target_device_user_id} onChange={v => set('target_device_user_id', v)} span />
          <Text   label="Pixel Format"    value={form.pixel_format}          onChange={v => set('pixel_format', v)}          hint="e.g. BayerRG8" />
          <Toggle label="ROI Enable"      value={form.roi_enable}            onChange={v => set('roi_enable', v)} />
          <Num    label="Width"           value={form.frame_width}           onChange={v => set('frame_width', v)}           suffix="px" />
          <Num    label="Height"          value={form.frame_height}          onChange={v => set('frame_height', v)}          suffix="px" />
          <Num    label="ROI Offset X"    value={form.roi_offset_x}          onChange={v => set('roi_offset_x', v)}          suffix="px" />
          <Num    label="ROI Offset Y"    value={form.roi_offset_y}          onChange={v => set('roi_offset_y', v)}          suffix="px" />
          <Toggle label="Auto Exposure"   value={form.auto_exposure}         onChange={v => set('auto_exposure', v)} />
          <Num    label="Exposure"        value={form.exposure_time_us}      onChange={v => set('exposure_time_us', v)}      suffix="µs" />
          <Toggle label="Auto Gain"       value={form.auto_gain}             onChange={v => set('auto_gain', v)} />
          <Num    label="Gain"            value={form.gain_db}               onChange={v => set('gain_db', v)}               suffix="dB" step="0.1" />
        </Card>

        {/* ── White Balance ── */}
        <Card title="White Balance" color="#8b5cf6">
          <Toggle label="Auto White Balance" value={form.white_balance_auto} onChange={v => set('white_balance_auto', v)} span />
          <Num    label="Red Ratio"    value={form.wb_red_ratio}   onChange={v => set('wb_red_ratio', v)}   step="0.1" />
          <Num    label="Green Ratio"  value={form.wb_green_ratio} onChange={v => set('wb_green_ratio', v)} step="0.1" />
          <Num    label="Blue Ratio"   value={form.wb_blue_ratio}  onChange={v => set('wb_blue_ratio', v)}  step="0.1" />
        </Card>

        {/* ── Engines & Processing ── */}
        <Card title="Engines & Processing" color="#10b981">
          <Num    label="Parallel Workers"    value={form.max_parallel_frames}      onChange={v => set('max_parallel_frames', v)} />
          <Toggle label="Template Matching"   value={form.enable_template_matching} onChange={v => set('enable_template_matching', v)} />
          <Toggle label="PatchCore CNN"       value={form.enable_patchcore}         onChange={v => set('enable_patchcore', v)} />
          <Text   label="PatchCore Model"     value={form.patchcore_model_path}     onChange={v => set('patchcore_model_path', v)} hint="e.g. 1.ckpt" span />
          <Num    label="PatchCore Threshold" value={form.patchcore_threshold}      onChange={v => set('patchcore_threshold', v)}  step="0.01" />
        </Card>

        {/* ── Output ── */}
        <Card title="Output" color="#64748b">
          <Text   label="Save Directory" value={form.save_directory} onChange={v => set('save_directory', v)} span />
          <Text   label="Format"         value={form.save_format}    onChange={v => set('save_format', v)}    hint="JPEG or BMP" />
          <Num    label="JPEG Quality"   value={form.jpeg_quality}   onChange={v => set('jpeg_quality', v)}   suffix="/ 100" />
        </Card>

        {/* ── Network ── */}
        <Card title="Network Config" color="#0ea5e9">
          <Text   label="Force IP"    value={form.target_force_ip}      onChange={v => set('target_force_ip', v)}      hint="169.x.x.x" />
          <Text   label="Subnet Mask" value={form.target_force_subnet}  onChange={v => set('target_force_subnet', v)}  hint="255.x.x.x" />
          <Text   label="Gateway"     value={form.target_force_gateway} onChange={v => set('target_force_gateway', v)} hint="169.x.x.x" span />
        </Card>

      </div>
    </div>
  )
}

/* ── Card ─────────────────────────────────────────── */
function Card({ title, color, children }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardHeader} style={{ borderLeftColor: color }}>
        <span className={styles.cardTitle}>{title}</span>
      </div>
      <div className={styles.fieldGrid}>{children}</div>
    </div>
  )
}

/* ── Field components ─────────────────────────────── */
function Text({ label, value, onChange, hint, span }) {
  return (
    <div className={`${styles.field} ${span ? styles.fieldSpan : ''}`}>
      <label className={styles.fieldLabel}>{label}</label>
      <input
        type="text"
        className={styles.input}
        value={value ?? ''}
        onChange={e => onChange(e.target.value)}
        placeholder={hint ?? ''}
      />
    </div>
  )
}

function Num({ label, value, onChange, suffix, step = '1', span }) {
  return (
    <div className={`${styles.field} ${span ? styles.fieldSpan : ''}`}>
      <label className={styles.fieldLabel}>{label}</label>
      <div className={styles.inputWrap}>
        <input
          type="number"
          className={`${styles.input} ${suffix ? styles.inputSuffixed : ''}`}
          value={value ?? ''}
          step={step}
          onChange={e => onChange(parseFloat(e.target.value))}
        />
        {suffix && <span className={styles.suffix}>{suffix}</span>}
      </div>
    </div>
  )
}

function Toggle({ label, value, onChange, span }) {
  return (
    <div className={`${styles.field} ${span ? styles.fieldSpan : ''}`}>
      <label className={styles.fieldLabel}>{label}</label>
      <div className={styles.toggleRow}>
        <button
          type="button"
          role="switch"
          aria-checked={!!value}
          className={`${styles.toggle} ${value ? styles.toggleOn : ''}`}
          onClick={() => onChange(!value)}
        >
          <span className={styles.thumb} />
        </button>
        <span className={styles.toggleLabel}>{value ? 'On' : 'Off'}</span>
      </div>
    </div>
  )
}
