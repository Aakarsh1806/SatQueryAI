import React, {useState} from 'react';
import {ArrowLeftRight, ArrowUpRight, Layers3, MessageSquare, ScanSearch, Trash2, Upload} from 'lucide-react';
import {Badge, Button, Card, Chip, Tooltip} from './ui';
import {exampleReason, fusionProblems, modality} from '../workspace';

export const specialists = [
  {id: 'rs_vlm', name: 'RS-VLM', model: 'Qwen2.5-VL-3B + LoRA', dataset: 'VRSBench', description: 'Answers questions and writes captions for optical satellite imagery.', kind: 'single', icon: MessageSquare, group: 'Ask', input: 'One optical image + question or caption request', output: 'QA & captions · text; spatial evidence not available', query: 'Describe this image', examples: ['Describe this image', 'How many buildings are visible?']},
  {id: 'grounding', name: 'Grounding', model: 'Grounding DINO Tiny', dataset: 'VRSBench', description: 'Locates objects described in text and returns bounding boxes.', kind: 'single', icon: ScanSearch, group: 'Locate', input: 'One optical image + object phrase', output: 'Object boxes and per-box scores · no masks', query: 'Where are the storage tanks?', examples: ['Where are the storage tanks?']},
  {id: 'change', name: 'Change', model: 'Chg2Cap', dataset: 'LEVIR-CC', description: 'Describes changes between before and after optical images.', kind: 'change', icon: ArrowLeftRight, group: 'Compare', input: 'Two optical images of the same area · A before, B after', output: 'Change captions · description only; no change map or question answering', query: 'What changed between these images?', examples: ['What changed between these images?']},
  {id: 'fusion', name: 'Fusion', model: 'Dual-encoder fusion network', dataset: 'SEN12MS summer subset', description: 'Classifies land cover using aligned optical and SAR imagery.', kind: 'fusion', icon: Layers3, group: 'Classify', input: 'Aligned optical + SAR GeoTIFFs · 13 + 2 bands, 256 × 256', output: 'Land cover · class + top 3 only; no free-form answers', query: 'Classify the land cover', examples: ['Classify the land cover']},
];

export function ModelSummary({model, id, compact = false, includeSpec = false}) {
  const spec = specialists.find(item => item.id === id);
  return <>{includeSpec && spec && <><p className="small"><strong>Input:</strong> {spec.input}</p><p className="small"><strong>Returns:</strong> {spec.output}</p></>}<div className="specialist-badges">
    {model ? <Badge tone={model.mode === 'real' ? 'info' : 'warning'}>{model.mode === 'real' ? 'Real' : 'Stub · synthetic output'}</Badge> : <Badge>Status unavailable</Badge>}
    {id === 'grounding' && <Badge tone="warning">Unvalidated</Badge>}
    {id === 'fusion' && <Badge tone="warning">Not yet verified on a real pair</Badge>}
  </div><p className="small validation-copy"><strong>Validation:</strong> {model?.validation_status || 'Status not available; model readiness has not been confirmed.'}</p>
    {!compact && model?.availability_note && <p className="small muted">{model.availability_note}</p>}
    {!compact && <p className="small muted">{model?.fallback_reason || (model?.mode === 'real' ? 'Real prerequisites found; this is not a quality-validation pass.' : model ? 'Configured stub; outputs are synthetic.' : 'Readiness could not be confirmed.')}</p>}</>;
}

export function HowItWorks({disabled, onChoose}) {
  return <Card className="how-it-works simplified-models"><h2>How it works</h2>
    <div className="specialist-grid">{specialists.map(({id, name, model, description, kind, query}) => <article key={id} className="specialist-card" onClick={event => {if (!disabled && !event.target.closest('button')) onChoose(kind, query);}}>
      <h3><button type="button" className="specialist-select" disabled={disabled} onClick={() => onChoose(kind, query)} aria-label={`Use ${name} example`}>{name}</button></h3>
      <p className="model-identity">{model}</p><p>{description}</p>
    </article>)}</div>
  </Card>;
}

export function InputRequirementsDetails() {
  return <section className="archived-input-requirements"><h2>Input requirements</h2>
      <h3>Single image</h3><p>One optical image for QA, captions or grounding. GeoTIFF, TIFF, PNG or JPG. Grounding returns boxes only, no masks, and is unvalidated. PNG/JPG have no georeferencing; CRS and footprint checks are skipped.</p>
      <h3>Before / After</h3><p>Change captions: upload two optical images of the same area, before then after. Images are prepared at 256 × 256. Returns a description only, with no change map or direct question answering. Confidence is not available.</p>
      <h3>Optical + SAR</h3><p>Real land-cover fusion requires a 13-band Sentinel-2 GeoTIFF (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12; raw digital numbers) and a 2-band Sentinel-1 GeoTIFF (VV, VH already in dB). Both must be exactly 256 × 256 with non-null identical CRS, transform and pixel grid. No resampling. Values must be finite. PNG/JPG or RGB files do not qualify.</p><p>Raw inputs are used for inference; RGB/SAR previews are display only. Pair alignment, band order and finite values are verified by the Agent. Fusion is an 8-class classifier, not free-form QA, and is not yet verified on a real pair. Stub or pre-computed placeholders are not live verified predictions.</p>
      <p>Routing stays automatic. Training-domain differences can degrade results; no evidence or confidence is guaranteed.</p>
  </section>;
}

function formatSize(bytes) {return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;}
function UploadSlot({index, slot, label, required, disabled, showFusionErrors, onFile, onOverride, onRemove, onReject}) {
  const [dragging, setDragging] = useState(false), [dropError, setDropError] = useState('');
  const id = `image-${index}`, problems = showFusionErrors ? fusionProblems(slot) : [];
  function choose(files) {
    setDragging(false); setDropError('');
    if (disabled || !files?.length) return;
    if (files.length > 1) return onReject('Add one image per slot; use the other slot for the second image.');
    if (!/\.(tiff?|png|jpe?g)$/i.test(files[0].name)) return onReject('Unsupported file type. Choose a GeoTIFF, TIFF, PNG or JPG image.');
    onFile(files[0]);
  }
  return <section className="upload-card" aria-labelledby={`${id}-title`}>
    <div className="row"><h3 id={`${id}-title`}>{label}</h3><span className="muted small">{required ? 'Required' : 'Optional'}</span></div>
    <label className={`dropzone ${slot.info ? 'has-image' : ''} ${dragging ? 'is-dragging' : ''} ${disabled ? 'is-disabled' : ''}`} htmlFor={id}
      onDragOver={event => {event.preventDefault(); if (!disabled) {event.dataTransfer.dropEffect = 'copy'; setDragging(true);}}}
      onDragLeave={event => {if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false);}}
      onDrop={event => {event.preventDefault(); choose(event.dataTransfer.files);}}>
      {slot.info?.preview_data_url ? <img className="thumbnail" src={slot.info.preview_data_url} alt={`Display-only ${modality(slot)} preview of ${slot.file?.name}`}/> : <span className="upload-symbol"><Upload size={18} strokeWidth={1.75} aria-hidden="true"/></span>}
      <strong>{slot.loading ? 'Inspecting image…' : slot.file?.name || 'Drop an image or browse'}</strong>
      <span className="small muted">{slot.file ? `${formatSize(slot.file.size)} · Click or drop to replace` : 'GeoTIFF, TIFF, PNG or JPG'}</span>
      {slot.info && <span className="preview-caption">Display preview only</span>}
      <input id={id} type="file" aria-label={`Choose ${label}`} aria-describedby={`${id}-feedback`} aria-invalid={!!(slot.error || dropError || problems.length)} accept=".tif,.tiff,.png,.jpg,.jpeg" disabled={disabled} onChange={event => {choose(event.target.files); event.target.value = '';}}/>
    </label>
    {slot.file && <div className="upload-info"><div className="row"><Badge tone={modality(slot) === 'sar' ? 'info' : 'neutral'}>{slot.info ? `${modality(slot) === 'sar' ? 'SAR' : 'Optical'} · ${slot.override ? 'override' : 'guess'}` : 'Not inspected'}</Badge><Button variant="ghost" icon={Trash2} aria-label={`Remove Image ${index + 1}`} disabled={disabled} onClick={() => {setDropError(''); onRemove();}}>Remove</Button></div>

    </div>}
    <div id={`${id}-feedback`}>{[dropError, slot.error, ...problems].filter(Boolean).map(message => <p className="error" role="alert" key={message}>{message}</p>)}</div>
  </section>;
}

export default function WorkspaceInputs({slots, inputType, query, setQuery, busy, submitting, issue, error, realFusion, onSubmit, onFile, onOverride, onRemove, onReject}) {
  const labels = inputType === 'change' ? ['Image 1 · Before (optical)', 'Image 2 · After (optical)'] : inputType === 'fusion' ? ['Image 1 · Optical / Sentinel-2', 'Image 2 · SAR / Sentinel-1'] : ['Image 1 · Optical', 'Image 2 · Additional image'];
  const actualFusion = slots.some(slot => slot.file && modality(slot) === 'sar') && slots.some(slot => slot.file && modality(slot) === 'optical');
  return <Card as="form" className="input-panel" onSubmit={onSubmit}>
    <div className="uploads">{slots.map((slot, index) => <UploadSlot key={index} index={index} slot={slot} label={labels[index]} required={index === 0 || inputType !== 'single'} disabled={busy} showFusionErrors={realFusion && (inputType === 'fusion' || actualFusion)} onFile={file => onFile(index, file)} onOverride={value => onOverride(index, value)} onRemove={() => onRemove(index)} onReject={message => onReject(index, message)}/>)}</div>
    <div className="query-heading"><label htmlFor="query"><h2>Your question or request</h2></label></div>
    <textarea id="query" maxLength="10000" placeholder="Ask about the scene, locate an object, or describe a change…" value={query} disabled={busy} onChange={event => setQuery(event.target.value)} rows="5"/>
    <section className="example-section" aria-labelledby="example-title"><h3 id="example-title">Try an example</h3><div className="example-groups">{specialists.map(({id, kind, group, icon: Icon, examples}) => {
      const reason = exampleReason(kind, slots);
      return <div key={id} className="example-group"><span className="eyebrow">{group}</span><div className="example-chips">{examples.map(example => {
        const chip = <Chip className={reason ? 'is-incompatible' : ''} disabled={busy} aria-label={example} onClick={() => setQuery(example)}><Icon size={16} strokeWidth={1.75} aria-hidden="true"/>{example}</Chip>;
        return reason ? <Tooltip content={reason} key={example}>{chip}</Tooltip> : <React.Fragment key={example}>{chip}</React.Fragment>;
      })}</div></div>;
    })}</div><p className="small muted">Muted examples need different inputs. You can still use their text; the Agent selects the specialist.</p></section>
    {error && <p className="error" role="alert">{error}</p>}
    <div className="run-analysis"><Button type="submit" variant="primary" loading={busy} disabled={!!issue} aria-describedby="run-reason">{submitting ? 'Uploading request…' : busy ? 'Analysis in progress…' : 'Run analysis'}{!busy && <ArrowUpRight size={18} strokeWidth={1.75} aria-hidden="true"/>}</Button><p id="run-reason" className="small muted" role="status">{busy ? 'The selected specialist is processing this request.' : issue || 'Ready. Pair alignment and final validation are checked by the Agent.'}</p></div>
  </Card>;
}
