'use client';

import { cn } from '@/lib/utils';
import { HTMLAttributes } from 'react';

interface GlassPanelProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'strong' | 'subtle';
  withNoise?: boolean;
}

export function GlassPanel({ 
  children, 
  className, 
  variant = 'default',
  withNoise = false,
  ...props 
}: GlassPanelProps) {
  const variantClasses = {
    default: 'glass',
    strong: 'glass-strong',
    subtle: 'bg-white/[0.015] border-white/5',
  };

  return (
    <div 
      className={cn(
        variantClasses[variant],
        'rounded-xl border p-6 transition-all duration-300',
        withNoise && 'texture-noise',
        className
      )} 
      {...props}
    >
      {children}
    </div>
  );
}
