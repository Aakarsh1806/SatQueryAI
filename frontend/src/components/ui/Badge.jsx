import React from 'react';
export function Badge({tone = 'neutral', className = '', children, ...props}) {
  return <span className={`ui-badge ui-badge--${tone} ${className}`} {...props}>{children}</span>;
}
export function Chip({tone = 'neutral', selected, onClick, disabled, className = '', children, ...props}) {
  if (!onClick) return <Badge tone={tone} className={className} {...props}>{children}</Badge>;
  return <button type="button" className={`ui-chip ui-badge--${tone} ${className}`} aria-pressed={selected} disabled={disabled} onClick={onClick} {...props}>{children}</button>;
}
