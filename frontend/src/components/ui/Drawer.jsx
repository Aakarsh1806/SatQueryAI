import React, { useEffect, useId, useRef } from 'react';
import { X } from 'lucide-react';
import { Button } from './Button';

export function Drawer({open, onClose, title, children, footer}) {
  const dialog = useRef(null); const heading = useRef(null); const id = useId();
  useEffect(() => {
    if (!open) return;
    const previousFocus = document.activeElement; const previousOverflow = document.body.style.overflow;
    const element = dialog.current;
    element.showModal(); heading.current?.focus(); document.body.style.overflow = 'hidden';
    return () => {element.close(); document.body.style.overflow = previousOverflow; if (previousFocus?.isConnected) previousFocus.focus();};
  }, [open]);
  return <dialog ref={dialog} className="ui-drawer" aria-labelledby={id} onCancel={event => {event.preventDefault(); onClose();}} onClick={event => {
    if (event.target !== dialog.current) return;
    const rect = dialog.current.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose();
  }}><div className="ui-drawer-content"><header className="ui-drawer-header"><h2 id={id} ref={heading} tabIndex={-1}>{title}</h2><Button variant="ghost" icon={X} aria-label={`Close ${title}`} onClick={onClose}/></header>
    <div className="ui-drawer-body">{children}</div>{footer && <footer className="ui-drawer-footer">{footer}</footer>}
  </div></dialog>;
}
