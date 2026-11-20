import React, {useEffect, useState} from 'react';
import {RefreshCw} from 'lucide-react';
import {modelNames} from '../api';
import {Badge, Button, Card, Skeleton} from './ui';
import {HowItWorks} from './WorkspaceInputs';
import EvidenceViewer from './EvidenceViewer';

function AnalysisResult({job, loading, pollError, retryPolling, onCancel, onRetry}) {
  const [now,setNow] = useState(Date.now());
  useEffect(() => {if (!loading) return; setNow(Date.now()); const timer = setInterval(() => setNow(Date.now()),1000); return () => clearInterval(timer);},[loading]);
  const result = job.result, metadata = result?.metadata;
  const chosen = modelNames[job.model_id || job.routing?.model_id] || metadata?.model_name || 'Analysis';
  const playback = metadata?.execution_type === 'pre-computed';
  const start = job.time && Date.parse(/[Z+]|-\d\d:\d\d$/.test(job.time.slice(10)) ? job.time : job.time+'Z');
  const elapsed = Number.isFinite(start) ? Math.max(0,Math.floor((now-start)/1000)) : null;
  return <Card className="result-panel simple-result" aria-busy={loading}>
    <header className="simple-result-heading"><h2>{chosen}</h2>{metadata && <Badge tone={playback || metadata.mode === 'stub' ? 'warning' : 'info'}>{playback ? 'Pre-computed · not live inference' : metadata.mode === 'stub' ? 'Stub · synthetic output' : 'Real model'}</Badge>}</header>
    {playback && metadata.demo_info?.placeholder && <p className="small muted">Placeholder output.</p>}
    {loading && <div className="simple-progress"><p role="status">{job.status === 'queued' ? 'Waiting to start…' : /download/i.test(job.progress_message || '') ? 'Downloading the model…' : /load|LoRA/i.test(job.progress_message || '') ? 'Loading the model…' : 'Generating response…'}</p><div className="row">{elapsed !== null && <span className="small muted">{elapsed} s elapsed</span>}<Button disabled={job.cancellation_requested} onClick={onCancel}>{job.cancellation_requested ? 'Cancelling…' : 'Cancel request'}</Button></div><Skeleton label="Waiting for the response" lines={3}/></div>}
    {pollError && <div className="error" role="alert"><p>{pollError}</p><Button icon={RefreshCw} onClick={retryPolling}>Resume polling</Button></div>}
    {(job.status === 'failed' || job.status === 'cancelled') && <section className={job.status === 'failed' ? 'error' : 'report-cancelled'} role={job.status === 'failed' ? 'alert' : 'status'}><h3>{job.status === 'cancelled' ? 'Request cancelled' : 'Analysis failed'}</h3><p>{job.input_explanation || job.error || 'No answer was produced.'}</p><Button icon={RefreshCw} onClick={onRetry}>Retry analysis</Button></section>}
    {result && <>
      <h3 className="simple-answer-title">{job.task === 'change_caption' ? 'Change description' : 'Answer'}</h3>
      {job.task === 'change_caption' && <div className="report-evidence"><EvidenceViewer job={job}/></div>}
      <p className="report-answer">{result.answer}</p>
      {job.task !== 'change_caption' && result.evidence && <section className="report-evidence" aria-label="Evidence"><EvidenceViewer job={job}/></section>}
    </>}
    {job.artifacts_expired && <p className="small muted retention-note">Images removed after the retention period. The stored result is retained.</p>}
  </Card>;
}
export default function ResultPanel(props) {
  if (!props.job && props.loading) return <Card className="result-panel"><h2>Uploading…</h2><Skeleton label="Uploading your images" lines={3}/></Card>;
  return props.job ? <AnalysisResult key={props.job.request_id} {...props}/> : <HowItWorks disabled={props.loading} onChoose={props.onChoose}/>;
}
