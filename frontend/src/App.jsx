import React, { useEffect, useRef, useState } from 'react';
import { api, analyzeForm, modelNames, pending } from './api';
import { ArrowLeft, ArrowRight, Cpu, History as HistoryIcon, Layers3, LayoutDashboard, RefreshCw } from 'lucide-react';
import { Badge, Button, Card, Drawer, EmptyState, Skeleton, Toast, Tooltip } from './components/ui';
import LimitationsView from './components/LimitationsView';
import WorkspaceInputs, {ModelSummary, InputRequirementsDetails, specialists} from './components/WorkspaceInputs';
import {inputIssue, obviousInputType} from './workspace';
import ResultPanel from './components/ResultPanel';

const emptySlot = () => ({file: null, info: null, override: '', loading: false, error: ''});
const formatTask = task => task?.replaceAll('_', ' ') || 'Awaiting routing';
const pageFromHash = () => location.hash === '#history' ? 'history' : location.hash === '#limitations' ? 'limitations' : 'workspace';


function History({onOpen, refreshKey}) {
  const [history, setHistory] = useState(null);
  const [offset, setOffset] = useState(0);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const controller = new AbortController(); setError(''); setHistory(null);
    api(`/api/requests?limit=10&offset=${offset}`, {signal: controller.signal}).then(setHistory).catch(error => {if (error.name !== 'AbortError') setError(error.message);});
    return () => controller.abort();
  }, [offset, refresh, refreshKey]);
  return <Card className="history"><div className="row"><div><span className="eyebrow">YOUR WORKSPACE</span><h1>Request history</h1></div><Button icon={RefreshCw} onClick={() => setRefresh(value => value + 1)}>Refresh</Button></div><p className="muted">Past analyses, their results, and execution traces.</p>{error && <p role="alert" className="error">{error}</p>}{!history && !error && <Skeleton label="Loading history…" lines={4}/>}{history && <>{history.requests.length ? <div className="history-list">{history.requests.map(job => <button type="button" className="history-row" key={job.request_id} onClick={() => onOpen(job)}><div><strong>{job.query}</strong><p className="small muted">{new Date(job.time.endsWith('Z') || /[+-]\d\d:\d\d$/.test(job.time) ? job.time : job.time + 'Z').toLocaleString()} · {modelNames[job.model_id] || formatTask(job.task)}</p>{job.artifacts_expired && <span className="small muted">Images expired · result retained</span>}</div>{job.status !== 'done' && <span className="history-status">{job.status}</span>}</button>)}</div> : <EmptyState title="No requests yet" description="Start an analysis to see it here." icon={HistoryIcon}/>}<div className="row pagination"><Button icon={ArrowLeft} disabled={offset === 0} onClick={() => setOffset(value => Math.max(0, value - 10))}>Previous</Button><span className="small muted">{history.total ? offset + 1 : 0}–{Math.min(offset + 10, history.total)} of {history.total}</span><Button icon={ArrowRight} disabled={offset + 10 >= history.total} onClick={() => setOffset(value => value + 10)}>Next</Button></div></>}</Card>;
}

export default function App() {
  const [page, setPage] = useState(pageFromHash);
  const [slots, setSlots] = useState([emptySlot(), emptySlot()]);
  const [query, setQuery] = useState('');
  const inputType = obviousInputType(slots) || 'single';
  const [job, setJob] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [pollError, setPollError] = useState('');
  const [pollRetry, setPollRetry] = useState(0);
  const [health, setHealth] = useState('Connecting');
  const [models, setModels] = useState([]);
  const [modelsError, setModelsError] = useState('');
  const [modelsOpen, setModelsOpen] = useState(false);
  const [lastLiveDevice, setLastLiveDevice] = useState(null);
  useEffect(() => {
    const metadata = job?.result?.metadata;
    if (metadata?.mode === 'real' && metadata.execution_type !== 'pre-computed') {
      const device = metadata.device?.split(':')[0];
      if (['cuda', 'mps', 'cpu'].includes(device)) setLastLiveDevice(device);
    }
  }, [job?.request_id, job?.result]);
  const targets = [...new Set(models.filter(model => model.mode === 'real').map(model => model.settings?.device).filter(device => ['cuda', 'mps', 'cpu'].includes(device)))];
  const deviceLabel = lastLiveDevice || (targets.length === 1 ? targets[0] : 'auto');
  const deviceNote = lastLiveDevice ? `Last live response used ${lastLiveDevice}. Models unload after requests; the API does not report an active loaded device.` : 'Configured selection only. Auto selects CUDA, MPS, then CPU. The actual device is shown after a live response; no active-device telemetry is available.';
  useEffect(() => {api('/api/models').then(data => setModels(data.models)).catch(error => setModelsError(error.message));}, []);
  const inspection = useRef([null, null]);
  const submitLock = useRef(false);
  const busy = submitting || pending(job?.status);
  const realFusion = models.find(model => model.id === 'fusion')?.mode === 'real';
  const issue = inputIssue(slots, query, inputType, realFusion);
  useEffect(() => {const handler = () => setPage(pageFromHash()); window.addEventListener('hashchange', handler); return () => window.removeEventListener('hashchange', handler);}, []);
  useEffect(() => {let live = true; api('/api/health').then(value => {if (live) setHealth(value.worker === 'running' ? 'Backend connected' : 'Worker stopped');}).catch(() => {if (live) setHealth('Backend unavailable');}); return () => {live = false;};}, []);
  useEffect(() => () => inspection.current.forEach(controller => controller?.abort()), []);
  const currentId = job?.request_id;
  const isPending = pending(job?.status);
  useEffect(() => {
    if (!currentId || !isPending) return;
    const controller = new AbortController(); let timer;
    setPollError('');
    async function poll() {
      try {const next = await api(`/api/requests/${currentId}`, {signal: controller.signal}); if (controller.signal.aborted) return; setJob(next); if (pending(next.status)) timer = setTimeout(poll, 900);}
      catch (error) {if (error.name !== 'AbortError') setPollError(error.message);}
    }
    poll();
    return () => {controller.abort(); clearTimeout(timer);};
  }, [currentId, isPending, pollRetry]);
  function updateSlot(index, patch) {setSlots(previous => previous.map((slot, i) => i === index ? {...slot, ...patch} : slot));}
  async function inspectFile(index, file, override = '') {
    inspection.current[index]?.abort();
    const controller = new AbortController(); inspection.current[index] = controller;
    updateSlot(index, {file, override, info: null, error: '', loading: true}); setError('');
    const form = new FormData(); form.append('file', file); if (override) form.append('modality_override', override);
    try {const info = await api('/api/inspect', {method: 'POST', body: form, signal: controller.signal}); if (!controller.signal.aborted) updateSlot(index, {info, loading: false});}
    catch (error) {if (error.name !== 'AbortError') updateSlot(index, {error: error.message, loading: false});}
  }
  async function submit(event, queryText = query.trim()) {
    event.preventDefault(); if (submitLock.current || busy) return;
    const validationIssue = inputIssue(slots, queryText, inputType, realFusion);
    if (validationIssue) return setError(validationIssue);
    submitLock.current = true; setSubmitting(true); setError(''); setPollError(''); setJob(null);
    try {const queued = await api('/api/analyze', {method: 'POST', body: analyzeForm(slots, queryText)}); setJob(queued);}
    catch (error) {setError(error.message);}
    finally {setSubmitting(false); submitLock.current = false;}
  }
  function retryAnalysis() {
    const retryQuery = job?.query || query;
    setQuery(retryQuery);
    if (!slots[0].file) {
      setError('Re-add the original input images before retrying this request.');
      document.getElementById('image-0')?.focus();
      return;
    }
    submit({preventDefault() {}}, retryQuery);
  }
  async function cancel() {
    try {setJob(await api(`/api/requests/${job.request_id}/cancel`, {method: 'POST'}));}
    catch (error) {setPollError(error.message);}
  }
  function openHistory(next) {setJob(next); setError(''); setPollError(''); location.hash = 'workspace'; setPage('workspace');}
  return <>
    <header className="header"><div className="header-inner">
      <a className="brand" href="#workspace"><img src="/logo-mark.svg" alt=""/><span>SatQuery <em>AI</em></span></a>
      <nav aria-label="Main navigation"><a className={page === 'workspace' ? 'active' : ''} aria-current={page === 'workspace' ? 'page' : undefined} href="#workspace"><LayoutDashboard size={18} strokeWidth={1.75} aria-hidden="true"/>Workspace</a><a className={page === 'history' ? 'active' : ''} aria-current={page === 'history' ? 'page' : undefined} href="#history"><HistoryIcon size={18} strokeWidth={1.75} aria-hidden="true"/>History</a></nav>
      <div className="status-cluster"><span className={`connection ${health === 'Backend connected' ? 'online' : ''}`}><i aria-hidden="true"/>{health}</span>
        <Tooltip content={deviceNote}><button className="device-status" type="button" aria-label={`Device: ${deviceLabel}${lastLiveDevice ? ', last live response' : ', configured selection'}`}><Cpu size={16} strokeWidth={1.75} aria-hidden="true"/>{deviceLabel}{lastLiveDevice && <span>· last</span>}</button></Tooltip>
        <Button icon={Layers3} aria-haspopup="dialog" aria-expanded={modelsOpen} onClick={() => setModelsOpen(true)}>Models</Button>
      </div>
    </div></header>
    <main>
      {modelsError && <Toast tone="danger" title="Model readiness could not be loaded">{modelsError}</Toast>}
      {page === 'limitations' ? <><LimitationsView/><Card className="limitations-reference"><InputRequirementsDetails/><details><summary>Model modes, validation and caveats</summary><p className="muted">Configured/effective modes and validation notes. Model availability is not evidence of accuracy.</p><p>Grounding is unvalidated: no trainer benchmark metrics. Model 4 is not yet verified on a real pair; aligned optical/SAR GeoTIFFs are required.</p><p>Evidence and confidence depend on the selected model. Scores are uncalibrated; unsupported evidence or confidence is not available.</p><p>All four real adapters are enabled when prerequisites are available. Model 4 is not yet verified on a real pair. Playback and stub fallbacks are explicitly labelled.</p>{models.map(model => <section key={model.id} className="model-reference"><h3>{model.display_name}</h3><p className="mono">{model.version}</p><ModelSummary includeSpec model={model} id={model.id}/></section>)}</details></Card></> : page === 'history' ? <History onOpen={openHistory} refreshKey={job?.status}/> : <>
        <div className="intro workspace-intro"><div><h1>Analysis workspace</h1></div><Badge tone="info">SIH26167 · College project</Badge></div>
        <div className="workspace"><WorkspaceInputs slots={slots} inputType={inputType} query={query} setQuery={setQuery} busy={busy} submitting={submitting} issue={issue} error={error} realFusion={realFusion} onSubmit={submit}
          onFile={inspectFile} onOverride={(index, value) => inspectFile(index, slots[index].file, value)} onReject={(index, message) => updateSlot(index, {error: message})}
          onRemove={index => {inspection.current[index]?.abort(); updateSlot(index, emptySlot());}}/>
          <ResultPanel onRetry={retryAnalysis} job={job} loading={busy} pollError={pollError} onCancel={cancel} retryPolling={() => setPollRetry(value => value + 1)} models={models} modelsError={modelsError} onChoose={(_kind, sample) => setQuery(sample)}/>
        </div>
      </>}
      <footer className="site-footer"><span>SatQuery AI - Academic demonstration - SIH26167</span></footer>
    </main>
    <Drawer open={modelsOpen} onClose={() => setModelsOpen(false)} title="Models">
      {modelsError && <Toast tone="danger" title="Readiness unavailable">{modelsError}</Toast>}
      {!models.length && !modelsError && <Skeleton label="Loading model status" lines={4}/>}
      <div className="model-status-list simplified-models">{specialists.map(spec => <Card key={spec.id}>
        <h3>{spec.model}</h3><p>{spec.description}</p><p className="model-training">Trained on {spec.dataset}.</p>
      </Card>)}</div>
    </Drawer>
  </>;
}
