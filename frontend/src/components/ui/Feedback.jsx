import React from 'react';
import { Info, ScanLine, X, TriangleAlert } from 'lucide-react';
import { Button } from './Button';
import { Card } from './Card';

export function Skeleton({className = '', label = 'Loading', lines = 1}) {
  return <div className={`ui-skeleton-group ${className}`} role="status"><span className="sr-only">{label}</span>{Array.from({length: lines}, (_, index) => <span className="ui-skeleton" aria-hidden="true" key={index}/>)}</div>;
}
export function Toast({tone = 'info', title, children, onDismiss}) {
  const Icon = tone === 'danger' || tone === 'warning' ? TriangleAlert : Info;
  return <div className={`ui-toast ui-badge--${tone}`} role={tone === 'danger' ? 'alert' : 'status'} onKeyDown={event => {if (event.key === 'Escape' && onDismiss) {event.stopPropagation(); onDismiss();}}}>
    <Icon size={18} strokeWidth={1.75} aria-hidden="true"/><div>{title && <strong>{title}</strong>}{children && <p>{children}</p>}</div>
    {onDismiss && <Button variant="ghost" icon={X} aria-label="Dismiss notification" onClick={onDismiss}/>}</div>;
}
export function EmptyState({title, description, icon: Icon = ScanLine, children, className = ''}) {
  return <Card className={`ui-empty-state ${className}`}><span className="ui-empty-icon"><Icon size={18} strokeWidth={1.75} aria-hidden="true"/></span><h2>{title}</h2><p className="muted">{description}</p>{children}</Card>;
}
