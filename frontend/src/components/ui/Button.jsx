import React, { forwardRef } from 'react';
import { LoaderCircle } from 'lucide-react';

export const Button = forwardRef(function Button({variant = 'secondary', loading = false, disabled = false, icon: Icon, children, className = '', type = 'button', ...props}, ref) {
  const Glyph = loading ? LoaderCircle : Icon;
  return <button ref={ref} type={type} className={`ui-button ui-button--${variant} ${!children ? 'ui-button--icon-only' : ''} ${className}`} disabled={disabled || loading} aria-busy={loading || undefined} {...props}>
    {Glyph && <Glyph size={18} strokeWidth={1.75} aria-hidden="true"/>}{children}
  </button>;
});
