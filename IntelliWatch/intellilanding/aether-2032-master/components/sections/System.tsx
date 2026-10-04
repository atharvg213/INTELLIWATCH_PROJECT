'use client';

import { motion } from 'framer-motion';
import { GlassPanel } from '@/components/ui/glass-panel';
import { ArrowUpRight } from 'lucide-react';

const principles = [
  {
    number: '01',
    title: 'Material Honesty',
    description: 'Every surface, every joint, every layer is expressed for what it is. No decoration. Only truth in the material.',
  },
  {
    number: '02',
    title: 'Layered Transparency',
    description: 'Depth is not simulated. It is constructed through multiple planes of glass, air, and structure — each with its own optical behavior.',
  },
  {
    number: '03',
    title: 'Silent Precision',
    description: 'Power that does not announce itself. Mechanisms operate at the edge of perception. The loudest statement is silence.',
  },
  {
    number: '04',
    title: 'Human Calibration',
    description: 'The system does not optimize for speed. It optimizes for intention. Every interaction is measured against the human hand and eye.',
  },
];

export function System() {
  return (
    <section id="system" className="relative border-t border-white/10 pt-20 pb-24">
      <div className="max-w-6xl mx-auto px-6">
        {/* Section Header */}
        <div className="max-w-2xl mb-16">
          <div className="text-[10px] tracking-[3px] text-[#666] mb-4">AETHER PROTOCOL</div>
          <h2 className="heading-section text-6xl tracking-[-1.5px] mb-6">
            The System
          </h2>
          <p className="text-xl text-[#a3a3a3] leading-tight">
            An architecture built on the belief that technology should feel 
            less like software and more like a precisely engineered instrument.
          </p>
        </div>

        {/* Principles Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {principles.map((principle, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: index * 0.05, ease: [0.23, 1, 0.32, 1] }}
            >
              <GlassPanel className="h-full group p-9 flex flex-col" withNoise>
                <div className="flex items-center justify-between mb-8">
                  <div className="font-mono text-[11px] tracking-[3px] text-[#666]">
                    {principle.number}
                  </div>
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity text-[#c14444]">
                    <ArrowUpRight size={18} />
                  </div>
                </div>

                <h3 className="text-3xl tracking-[-0.5px] mb-4 font-light">
                  {principle.title}
                </h3>

                <p className="text-[#a3a3a3] text-[15px] leading-snug flex-1">
                  {principle.description}
                </p>
              </GlassPanel>
            </motion.div>
          ))}
        </div>

        {/* Bottom statement */}
        <div className="mt-16 max-w-xl">
          <p className="text-[#666] text-sm leading-relaxed tracking-wide">
            The system does not evolve through features. It evolves through refinement. 
            Every generation removes what is unnecessary until only intention remains.
          </p>
        </div>
      </div>
    </section>
  );
}
