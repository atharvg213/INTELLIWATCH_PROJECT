'use client';

import { GlassPanel } from '@/components/ui/glass-panel';

export function Closing() {
  return (
    <section className="relative border-t border-white/10 pt-20 pb-16">
      <div className="max-w-4xl mx-auto px-6 text-center">
        <div className="max-w-xl mx-auto">
          <h2 className="heading-section text-5xl md:text-6xl tracking-[-1.2px] mb-6">
            Built for those who notice.
          </h2>
          <p className="text-xl text-[#a3a3a3] mb-10">
            AETHER is not for everyone. It is for those who believe that the highest form of technology is one that feels inevitable.
          </p>
        </div>

        <button 
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="btn btn-primary text-sm px-10 py-4"
        >
          Begin a conversation
        </button>
      </div>

      {/* Minimal Footer */}
      <footer className="mt-24 pt-8 border-t border-white/10">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row md:items-center justify-between gap-y-4 text-xs tracking-[1px] text-[#666]">
          <div>© AETHER 2032</div>
          <div className="flex gap-x-6">
            <a href="#" className="hover:text-white transition-colors">Protocol</a>
            <a href="#" className="hover:text-white transition-colors">Archive</a>
            <a href="#" className="hover:text-white transition-colors">Contact</a>
            <a href="#" className="hover:text-white transition-colors">Legal</a>
          </div>
          <div className="text-[#444]">Precision is the only luxury left.</div>
        </div>
      </footer>
    </section>
  );
}
