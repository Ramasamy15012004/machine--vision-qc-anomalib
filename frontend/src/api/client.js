const BASE = 'http://localhost:8000'

export const startSystem = () =>
  fetch(`${BASE}/start`, { method: 'POST' }).then(r => r.json())

export const stopSystem = () =>
  fetch(`${BASE}/stop`, { method: 'POST' }).then(r => r.json())

export const getStatus = () =>
  fetch(`${BASE}/status`).then(r => r.json())

export const getDetected = () =>
  fetch(`${BASE}/detected`).then(r => r.json())

export const liveProcessUrl = () => `${BASE}/live_process`

export const getConfig = () =>
  fetch(`${BASE}/config`).then(r => r.json())

export const saveConfig = (data) =>
  fetch(`${BASE}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  }).then(r => r.json())
