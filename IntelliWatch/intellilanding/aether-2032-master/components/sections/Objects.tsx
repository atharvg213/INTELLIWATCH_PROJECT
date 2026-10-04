'use client';

import { motion } from 'framer-motion';
import { GlassPanel } from '@/components/ui/glass-panel';

const objects = [
  {
    name: 'A1 Core Module',
    type: 'Precision Instrument',
    desc: 'The foundational unit. A transparent vessel containing calibrated mechanisms that respond to human intent.',
  },
  {
    name: 'A2 Spatial Node',
    type: 'Environmental Interface',
    desc: 'A floating architectural element. Light, sound, and presence are sensed and answered without screens.',
  },
  {
    name: 'A3 Personal Anchor',
    type: 'Wearable System',
    desc: 'An object that disappears on the body. It measures, protects, and connects through material intelligence.',
  },
];

export function Objects() {
  return (
    <section id="objects" className="relative border-t border-white/10 pt-20 pb-24">
      <div className="max-w-6xl mx-auto px-6">
        <div className="mb-14">
          <div className="text-[10px] tracking-[3px] text-[#666] mb-3">ARTIFACTS</div>
          <h2 className="heading-section text-6xl tracking-[-1.5px]">Objects</h2>
        </div>

        <div className="space-y-6">
          {objects.map((obj, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.04 }}
            >
              <GlassPanel className="group p-10 md:p-12 flex flex-col md:flex-row md:items-center gap-8 md:gap-16" withNoise>
                <div className="md:w-1/3">
                  <div className="font-mono text-xs tracking-[2px] text-[#c14444] mb-2">{obj.type}</div>
                  <h3 className="text-4xl tracking-[-0.8px] font-light">{obj.name}</h3>
                </div>
                <div className="md:w-2/3 text-lg text-[#a3a3a3] leading-tight">
                  {obj.desc}
                </div>
                <div className="md:ml-auto text-[#666] group-hover:text-white transition-colors text-sm tracking-widest">
                  SPEC →
                </div>
              </GlassPanel>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
