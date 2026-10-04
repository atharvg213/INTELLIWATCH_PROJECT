'use client';

import { motion } from 'framer-motion';
import { GlassPanel } from '@/components/ui/glass-panel';

const projects = [
  {
    year: '2031',
    title: 'K2 Observatory',
    location: 'Swiss Alps',
    desc: 'A silent observation platform for high-altitude research. Zero screens. Pure material presence.',
    role: 'Complete system design & fabrication',
  },
  {
    year: '2030',
    title: 'Lumen Archive',
    location: 'Reykjavik, Iceland',
    desc: 'A light-sensitive archive that protects knowledge through calibrated glass and stone.',
    role: 'Architecture + interaction system',
  },
  {
    year: '2029',
    title: 'Vesper',
    location: 'Atacama Desert',
    desc: 'A mobile precision instrument for nocturnal observation. Designed to disappear into the landscape.',
    role: 'Product + software architecture',
  },
];

export function Showcase() {
  return (
    <section id="work" className="relative border-t border-white/10 pt-20 pb-24 bg-[#050505]">
      <div className="max-w-6xl mx-auto px-6">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between mb-14 gap-6">
          <div>
            <div className="text-[10px] tracking-[3px] text-[#666] mb-3">SELECTED WORK</div>
            <h2 className="heading-section text-6xl tracking-[-1.5px]">Archive</h2>
          </div>
          <p className="max-w-sm text-[#a3a3a3] text-lg">
            Projects where technology becomes invisible and only intention remains.
          </p>
        </div>

        <div className="space-y-6">
          {projects.map((project, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.05 }}
            >
              <GlassPanel className="group p-10 md:p-12" withNoise>
                <div className="flex flex-col md:flex-row md:items-center gap-8 md:gap-16">
                  <div className="md:w-1/4">
                    <div className="font-mono text-xs tracking-[2px] text-[#c14444] mb-1">{project.year}</div>
                    <h3 className="text-4xl tracking-[-0.6px] font-light mb-2">{project.title}</h3>
                    <div className="text-sm text-[#666]">{project.location}</div>
                  </div>

                  <div className="md:w-2/4 text-lg text-[#a3a3a3] leading-tight">
                    {project.desc}
                  </div>

                  <div className="md:w-1/4 md:text-right text-sm text-[#666] group-hover:text-white transition-colors">
                    {project.role} <span className="inline-block group-hover:translate-x-1 transition-transform">→</span>
                  </div>
                </div>
              </GlassPanel>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
