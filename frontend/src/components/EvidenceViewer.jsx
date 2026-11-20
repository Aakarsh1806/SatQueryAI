import React, {useState} from 'react';
import {RefreshCw} from 'lucide-react';
import {apiUrl} from '../api';
import {Button, SegmentedControl} from './ui';

function Image({url, alt, onLoad, className = ''}) {
  const [failed, setFailed] = useState(false), [attempt, setAttempt] = useState(0);
  return failed ? <div className="image-unavailable"><p className="small muted">Image could not be loaded. The stored answer remains available.</p><Button icon={RefreshCw} onClick={() => {setFailed(false); setAttempt(value => value + 1);}}>Retry image</Button></div> : <img key={`${url}-${attempt}`} src={apiUrl(url)} alt={alt} className={className} onLoad={onLoad} onError={() => setFailed(true)}/>;
}
function Grounding({job, trace}) {
  const evidence = job.result.evidence, [active, setActive] = useState(null), [size, setSize] = useState(null);
  const url = !job.artifacts_expired && (job.preview_urls?.[0] || job.evidence_overlay_urls?.[0]);
  const info = trace?.validation?.images?.[0];
  const width = info?.width || size?.width, height = info?.height || size?.height;
  return <div><h3>Detected objects</h3>
    {!evidence?.boxes?.length ? <p className="small muted">{Array.isArray(evidence?.boxes) ? 'No objects were detected.' : 'Detection image not available.'}</p> : <div className="grounding-view">
      {url ? <div className="box-canvas"><Image url={url} alt="Input image with candidate grounding boxes" onLoad={event => setSize({width:event.currentTarget.naturalWidth,height:event.currentTarget.naturalHeight})}/>
        {width && height && <svg viewBox={`0 0 ${width} ${height}`} className="box-overlay" aria-hidden="true">{evidence.boxes.map(([x1,y1,x2,y2],index) => <g key={index} className={active === index ? 'is-highlighted' : ''}><rect x={x1} y={y1} width={Math.max(0,x2-x1)} height={Math.max(0,y2-y1)} vectorEffect="non-scaling-stroke"/><text x={x1+3} y={Math.max(14,y1-4)} fontSize={Math.max(12,width/40)}>{index+1}</text></g>)}</svg>}
      </div> : <p className="small muted">Image not available. Detected object labels remain below.</p>}
      <ol className="box-list" aria-label="Detected boxes">{evidence.boxes.map((box,index) => <li key={index}><button type="button" aria-pressed={active === index} onMouseEnter={() => setActive(index)} onMouseLeave={() => setActive(null)} onFocus={() => setActive(index)} onBlur={() => setActive(null)} onClick={() => setActive(index)}>
        <span className="box-list-title"><strong>{index+1}. {evidence.labels?.[index] || 'Candidate region'}</strong></span>
      </button></li>)}</ol>
    </div>}
  </div>;
}
function Change({job}) {
  const [view,setView] = useState('side'), [position,setPosition] = useState(50);
  const urls = job.artifacts_expired ? [] : job.preview_urls || [];
  return <div><div className="evidence-heading"><h3>Before / after inputs</h3><SegmentedControl label="Comparison view" value={view} onChange={setView} options={[{value:'side',label:'Side by side'},{value:'slider',label:'Slider'}]}/></div>
    {urls.length === 2 ? view === 'side' ? <div className="change-side"><figure><Image url={urls[0]} alt="Before image A, prepared input"/><figcaption>A · Before</figcaption></figure><figure><Image url={urls[1]} alt="After image B, prepared input"/><figcaption>B · After</figcaption></figure></div> : <div className="comparison"><div className="comparison-images"><Image url={urls[1]} alt="After image B, prepared input"/><div className="before-layer" style={{clipPath:`inset(0 ${100-position}% 0 0)`}}><Image url={urls[0]} alt="Before image A, prepared input"/></div><div className="divider" style={{left:`${position}%`}}/><span className="image-label left">A · Before</span><span className="image-label right">B · After</span></div><label className="slider-label">Before / after comparison<input type="range" aria-label="Before / after comparison" min="0" max="100" value={position} onChange={event => setPosition(Number(event.target.value))}/></label></div> : <p className="small muted">Input images: not available.</p>}
  </div>;
}
export default function EvidenceViewer({job, trace}) {
  if (job.task === 'change_caption') return <Change job={job}/>;
  const evidence = job.result?.evidence;
  if (job.task === 'grounding' || evidence?.boxes) return <Grounding job={job} trace={trace}/>;
  if (evidence?.top3) return <div><h3>Top 3 classes</h3><ol className="class-labels">{evidence.top3.map(([label]) => <li key={label}>{label}</li>)}</ol></div>;
  return null;
}
