'use client';

import { motion } from 'framer-motion';
import { GlassPanel } from '@/components/ui/glass-panel';

const steps = [
  {
    phase: '01',
    title: 'Observation',
    desc: 'We begin by studying the physical world at the scale of the hand and the eye. No assumptions.',
  },
  {
    phase: '02',
    title: 'Reduction',
    desc: 'Every element is questioned until only what is necessary remains. Complexity is earned, not added.',
  },
  {
    phase: '03',
    title: 'Materialization',
    desc: 'Ideas become precise instruments. Prototypes are built by hand before any digital rendering.',
  },
  {
    phase: '04',
    title: 'Calibration',
    desc: 'The final object is tuned until it disappears into the user’s intention. Nothing feels like technology.',
  },
];

export function Process() {
  return (
    <section id="process" className="relative border-t border-white/10 pt-20 pb-24 bg-[#050505]">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between mb-14 gap-6">
          <div>
            <div className="text-[10px] tracking-[3px] text-[#666] mb-3">METHOD</div>
            <h2 className="heading-section text-6xl tracking-[-1.5px]">Process</h2>
          </div>
          <p className="max-w-sm text-[#a3a3a3] text-lg">
            A four-stage discipline that treats design as engineering and engineering as care.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.06, ease: [0.23, 1, 0.32, 1] }}
            >
              <GlassPanel className="h-full p-8 flex flex-col" withNoise>
                <div className="font-mono text-xs tracking-[3px] text-[#c14444] mb-6">
                  {step.phase}
                </div>
                <h3 className="text-3xl tracking-[-0.4px] mb-4 font-light">{step.title}</h3>
                <p className="text-[#a3a3a3] text-[15px] leading-snug mt-auto">
                  {step.desc}
                </p>
              </GlassPanel>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
