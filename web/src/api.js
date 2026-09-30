// Default the API host to the same machine the page is served from, so opening
// the site on a phone (http://<LAN-IP>:5173) talks to the backend at :8000.
const API =
  import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`

let token = localStorage.getItem('cs_token') || null

export function getUsername() {
  return localStorage.getItem('cs_user') || null
}
export function isAuthed() {
  return !!token
}
export function logout() {
  token = null
  localStorage.removeItem('cs_token')
  localStorage.removeItem('cs_user')
}
function setAuth(t, username) {
  token = t
  localStorage.setItem('cs_token', t)
  localStorage.setItem('cs_user', username)
}

async function req(path, { method = 'GET', body, form } = {}) {
  const headers = {}
  if (token) headers.Authorization = `Bearer ${token}`
  let payload
  if (form) {
    payload = form
  } else if (body) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }
  const res = await fetch(`${API}${path}`, { method, headers, body: payload })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export const api = {
  photoUrl: (name) => `${API}/photos/${name}`,

  async register(username, password) {
    const r = await req('/auth/register', { method: 'POST', body: { username, password } })
    setAuth(r.token, r.username)
    return r
  },
  async login(username, password) {
    const r = await req('/auth/login', { method: 'POST', body: { username, password } })
    setAuth(r.token, r.username)
    return r
  },
  identify(file) {
    const fd = new FormData()
    fd.append('file', file)
    return req('/identify', { method: 'POST', form: fd })
  },
  addToCollection(item) {
    return req('/collection', { method: 'POST', body: item })
  },
  getCollection() {
    return req('/collection')
  },
  submitUnknown(label) {
    return req('/unknown', { method: 'POST', body: label })
  },
}
