'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { GlassPanel } from '@/components/ui/glass-panel';

const modes = [
  { id: 'precision', label: 'Precision', desc: 'Maximum mechanical feedback. Every movement is registered.' },
  { id: 'presence', label: 'Presence', desc: 'The system responds only to human proximity and breath.' },
  { id: 'silence', label: 'Silence', desc: 'All output is reduced to the absolute minimum required.' },
];

export function Experience() {
  const [activeMode, setActiveMode] = useState('precision');

  return (
    <section id="experience" className="relative border-t border-white/10 pt-20 pb-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="max-w-2xl mb-14">
          <div className="text-[10px] tracking-[3px] text-[#666] mb-3">INTERACTION</div>
          <h2 className="heading-section text-6xl tracking-[-1.5px]">Experience</h2>
          <p className="text-xl text-[#a3a3a3] mt-4">
            The interface should not be felt. Only the result.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Mode Selector */}
          <div className="lg:col-span-2 space-y-3">
            {modes.map((mode) => (
              <button
                key={mode.id}
                onClick={() => setActiveMode(mode.id)}
                className={`w-full text-left p-8 rounded-xl border transition-all duration-300 ${
                  activeMode === mode.id 
                    ? 'glass border-white/20' 
                    : 'border-white/5 hover:border-white/10'
                }`}
              >
                <div className="text-2xl tracking-[-0.4px] mb-3">{mode.label}</div>
                <div className="text-sm text-[#a3a3a3]">{mode.desc}</div>
              </button>
            ))}
          </div>

          {/* Visual Response Area */}
          <div className="lg:col-span-3">
            <GlassPanel className="h-full min-h-[420px] flex items-center justify-center relative overflow-hidden" withNoise>
              <div className="text-center">
                <div className="font-mono text-xs tracking-[3px] text-[#666] mb-4">LIVE SIMULATION</div>
                
                <motion.div
                  key={activeMode}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
                  className="text-7xl md:text-8xl tracking-[-3px] font-light mb-6"
                >
                  {activeMode === 'precision' && '0.008mm'}
                  {activeMode === 'presence' && 'Human'}
                  {activeMode === 'silence' && '—'}
                </motion.div>

                <p className="max-w-xs mx-auto text-[#a3a3a3]">
                  {activeMode === 'precision' && 'The system registers movement at the edge of human perception.'}
                  {activeMode === 'presence' && 'Nothing moves until a human enters the field.'}
                  {activeMode === 'silence' && 'The quietest possible response. Almost nothing.'}
                </p>
              </div>

              {/* Subtle animated elements */}
              <div className="absolute inset-0 pointer-events-none">
                {activeMode === 'precision' && (
                  <div className="absolute inset-0 bg-[radial-gradient(#c14444_0.6px,transparent_1px)] bg-[length:4px_4px] opacity-10" />
                )}
                {activeMode === 'presence' && (
                  <motion.div 
                    className="absolute inset-0 border border-white/5 rounded-full"
                    animate={{ scale: [1, 1.08, 1] }}
                    transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
                  />
                )}
              </div>
            </GlassPanel>
          </div>
        </div>
      </div>
    </section>
  );
}
