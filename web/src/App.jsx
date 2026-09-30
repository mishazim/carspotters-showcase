import { useEffect, useRef, useState } from 'react'
import { api, isAuthed, getUsername, logout } from './api.js'

const TIER_LABEL = {
  grey: 'Common', green: 'Uncommon', blue: 'Rare',
  purple: 'Epic', orange: 'Legendary', mythic: 'Mythic', unknown: 'Unknown',
}

export default function App() {
  const [authed, setAuthed] = useState(isAuthed())
  const [tab, setTab] = useState('spot')

  if (!authed) return <Auth onAuthed={() => setAuthed(true)} />

  return (
    <div className="app">
      <header>
        <h1>🚗 CarSpotters</h1>
        <button className="link" onClick={() => { logout(); setAuthed(false) }}>
          {getUsername()} · log out
        </button>
      </header>

      {tab === 'spot' ? <Spot /> : <Collection />}

      <nav className="tabs">
        <button className={tab === 'spot' ? 'on' : ''} onClick={() => setTab('spot')}>📸 Spot</button>
        <button className={tab === 'collection' ? 'on' : ''} onClick={() => setTab('collection')}>📔 CarDex</button>
      </nav>
    </div>
  )
}

function Auth({ onAuthed }) {
  const [mode, setMode] = useState('login')
  const [username, setU] = useState('')
  const [password, setP] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setErr(''); setBusy(true)
    try {
      await (mode === 'login' ? api.login(username, password) : api.register(username, password))
      onAuthed()
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="app center">
      <h1>🚗 CarSpotters</h1>
      <p className="muted">Spot cars in the wild. Build your CarDex.</p>
      <form onSubmit={submit} className="card-form">
        <input placeholder="username" value={username} onChange={e => setU(e.target.value)} autoCapitalize="none" />
        <input placeholder="password" type="password" value={password} onChange={e => setP(e.target.value)} />
        {err && <div className="err">{err}</div>}
        <button disabled={busy || !username || !password}>
          {busy ? '…' : mode === 'login' ? 'Log in' : 'Create account'}
        </button>
      </form>
      <button className="link" onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setErr('') }}>
        {mode === 'login' ? 'Need an account? Sign up' : 'Have an account? Log in'}
      </button>
    </div>
  )
}

function Spot() {
  const [preview, setPreview] = useState(null)
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [msgOk, setMsgOk] = useState(false)
  const inputRef = useRef()

  function pick(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null); setMsg('')
  }

  async function identify() {
    setBusy(true); setMsg('')
    try { setResult(await api.identify(file)) }
    catch (e) { setMsg(e.message); setMsgOk(false) } finally { setBusy(false) }
  }

  // Clear the photo + result so the user can scan another vehicle.
  // (Keeps any toast message so it stays visible above the dropzone.)
  function reset() {
    setPreview(null); setFile(null); setResult(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  function done(m) { setMsg(m); setMsgOk(true); reset() }

  return (
    <main className="spot">
      {msg && <div className={msgOk ? 'note' : 'err'}>{msg}</div>}

      {!preview && (
        <label className="dropzone">
          <input ref={inputRef} type="file" accept="image/*" capture="environment" onChange={pick} hidden />
          <div className="big">📷</div>
          <div>Take or upload a car photo</div>
        </label>
      )}

      {preview && (
        <>
          <img className="preview" src={preview} alt="car" />
          {!result && (
            <div className="row">
              <button className="ghost" onClick={reset}>Retake</button>
              <button onClick={identify} disabled={busy}>{busy ? 'Identifying…' : 'Identify'}</button>
            </div>
          )}
        </>
      )}

      {result && (
        <>
          {result.recognized
            ? <Card result={result} onDone={done} />
            : <UnknownForm result={result} onDone={done} />}
          <MatchDetails result={result} />
          <button className="ghost wide" onClick={reset}>↺ Scan another vehicle</button>
        </>
      )}
    </main>
  )
}

// Dev-only panel: shows the model's confidence for every scan (recognized or
// not), with the full top-3. Hidden automatically in production builds.
function MatchDetails({ result }) {
  if (!import.meta.env.DEV) return null
  return (
    <div className="match-details">
      <div className="md-head">
        🔍 Match details (dev) · {result.recognized
          ? `recognized (${Math.round(result.confidence * 100)}%)`
          : `below ${35}% threshold → label flow`}
      </div>
      <ul>
        {result.top_guesses?.map((g, i) => (
          <li key={i}>
            <span className="md-label">{g.label}</span>
            <span className="md-bar"><span style={{ width: `${Math.min(100, g.confidence * 100)}%` }} /></span>
            <span className="md-pct">{(g.confidence * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Card({ result, onDone }) {
  const [busy, setBusy] = useState(false)
  const tier = result.rarity_tier || 'unknown'
  // An exact catalog hit is "confident"; a make-only fallback is not, so we
  // proactively invite a correction in that case.
  const inexact = result.matched_by !== 'catalog'
  const [correcting, setCorrecting] = useState(inexact)
  const [make, setMake] = useState(result.make || '')
  const [model, setModel] = useState(result.model || '')
  const [year, setYear] = useState('')

  async function add() {
    setBusy(true)
    try {
      await api.addToCollection({
        photo_token: result.photo_token,
        make: result.make, model: result.model,
        rarity_tier: tier, confidence: result.confidence,
      })
      onDone('Added to your CarDex! +10 XP')
    } catch (e) { onDone(e.message) } finally { setBusy(false) }
  }

  // A correction also feeds the training flywheel (predicted vs. corrected).
  async function addCorrected(e) {
    e.preventDefault()
    setBusy(true)
    try {
      const res = await api.submitUnknown({
        photo_token: result.photo_token,
        predicted_label: result.label,
        predicted_confidence: result.confidence,
        make, model, year: year ? Number(year) : null,
      })
      onDone(res.message || 'Saved your correction!')
    } catch (e) { onDone(e.message) } finally { setBusy(false) }
  }

  return (
    <div className={`card tier-${tier}`}>
      <div className="tier-badge">{TIER_LABEL[tier]}</div>
      <h2>{result.make} {result.model}</h2>
      <div className="muted">{Math.round(result.confidence * 100)}% confident
        {result.matched_by === 'make_default' && ' · rarity by make (model not in catalog)'}</div>
      <button onClick={add} disabled={busy}>{busy ? '…' : `Add to Collection`}</button>

      {!correcting && (
        <button type="button" className="link" onClick={() => setCorrecting(true)}>
          Not an exact match? Correct it
        </button>
      )}
      {correcting && (
        <form className="correction" onSubmit={addCorrected}>
          <div className="muted">Not an exact match? Enter the correct make, model, and year below:</div>
          <input placeholder="Make" value={make} onChange={e => setMake(e.target.value)} />
          <input placeholder="Model" value={model} onChange={e => setModel(e.target.value)} />
          <input placeholder="Year (optional)" inputMode="numeric" value={year} onChange={e => setYear(e.target.value)} />
          <button className="ghost" disabled={busy || !make || !model}>{busy ? '…' : 'Add corrected car'}</button>
        </form>
      )}
    </div>
  )
}

function UnknownForm({ result, onDone }) {
  const [make, setMake] = useState('')
  const [model, setModel] = useState('')
  const [year, setYear] = useState('')
  const [busy, setBusy] = useState(false)
  const guess = result.top_guesses?.[0]

  async function submit(e) {
    e.preventDefault(); setBusy(true)
    try {
      const res = await api.submitUnknown({
        photo_token: result.photo_token,
        predicted_label: guess?.label, predicted_confidence: guess?.confidence,
        make, model, year: year ? Number(year) : null,
      })
      onDone(res.message || 'Thanks! Your label helps train the model.')
    } catch (e) { onDone(e.message) } finally { setBusy(false) }
  }

  return (
    <form className="card tier-unknown" onSubmit={submit}>
      <div className="tier-badge">Not recognized</div>
      <p className="muted">
        This car isn't in our dataset yet{guess && ` (closest guess: ${guess.label})`}.
        Label it to help the model learn!
      </p>
      <input placeholder="Make (e.g. Toyota)" value={make} onChange={e => setMake(e.target.value)} />
      <input placeholder="Model (e.g. Supra)" value={model} onChange={e => setModel(e.target.value)} />
      <input placeholder="Year (optional)" inputMode="numeric" value={year} onChange={e => setYear(e.target.value)} />
      <button disabled={busy || !make || !model}>{busy ? '…' : 'Submit label'}</button>
    </form>
  )
}

function Collection() {
  const [cars, setCars] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    api.getCollection().then(r => setCars(r.cars)).catch(e => setErr(e.message))
  }, [])

  if (err) return <main><div className="err">{err}</div></main>
  if (!cars) return <main><p className="muted">Loading…</p></main>
  if (!cars.length) return <main className="center"><p className="muted">No cars yet. Go spot some!</p></main>

  return (
    <main>
      <p className="muted">{cars.length} car{cars.length !== 1 && 's'} collected</p>
      <div className="grid">
        {cars.map(c => (
          <div key={c.id} className={`dex tier-${c.rarity_tier || 'unknown'}`}>
            <img src={api.photoUrl(c.photo_path)} alt={c.model} />
            <div className="dex-body">
              <strong>{c.make} {c.model}</strong>
              <span className="tier-badge sm">{TIER_LABEL[c.rarity_tier] || 'Unknown'}</span>
            </div>
          </div>
        ))}
      </div>
    </main>
  )
}
