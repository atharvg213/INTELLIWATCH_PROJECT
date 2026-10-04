"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Menu } from "lucide-react";
import dynamic from "next/dynamic";
import { System } from "@/components/sections/System";
import { Process } from "@/components/sections/Process";
import { Objects } from "@/components/sections/Objects";
import { Showcase } from "@/components/sections/Showcase";
import { Experience } from "@/components/sections/Experience";
import { Closing } from "@/components/sections/Closing";

/**
 * AETHER — 2032
 * Industrial technology with human precision.
 * Nothing geometry + Liquid Glass depth.
 */

// Dynamically import the heavy 3D component (no SSR)
const FloatingCore = dynamic(
  () => import("@/components/three/FloatingCore"),
  { ssr: false }
);

export default function AetherIndustrial() {
  const [menuOpen, setMenuOpen] = useState(false);

  const navLinks = [
    { label: 'System', href: '#system' },
    { label: 'Process', href: '#process' },
    { label: 'Work', href: '#work' },
    { label: 'Experience', href: '#experience' },
    { label: 'Objects', href: '#objects' },
  ];

  const scrollTo = (href: string) => {
    const element = document.querySelector(href);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    setMenuOpen(false);
  };

  return (
    <div className="min-h-screen bg-[#050505] text-[#f5f5f5] selection:bg-white selection:text-black">
      {/* Minimal Floating Navigation */}
      <nav className="fixed top-6 left-1/2 -translate-x-1/2 z-50 w-[92%] max-w-6xl">
        <div className="glass flex items-center justify-between px-6 py-3.5 rounded-xl border border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-[#c14444]" />
            <div className="font-mono text-[11px] tracking-[3px] text-[#a3a3a3]">AETHER</div>
            <div className="text-[10px] text-[#666] font-mono pl-1">2032</div>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8 text-sm tracking-[0.5px]">
            {navLinks.map((link) => (
              <button 
                key={link.href}
                onClick={() => scrollTo(link.href)}
                className="hover:text-white/70 transition-colors"
              >
                {link.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3">
            <button 
              onClick={() => scrollTo('#system')}
              className="btn hidden md:block text-xs px-5 py-2"
            >
              Request Access
            </button>
            <button 
              onClick={() => setMenuOpen(!menuOpen)}
              className="md:hidden p-2"
            >
              <Menu size={18} />
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {menuOpen && (
          <div className="md:hidden absolute top-20 left-1/2 -translate-x-1/2 w-[92%] max-w-6xl glass border border-white/10 rounded-xl p-6 flex flex-col gap-4 text-sm">
            {navLinks.map((link) => (
              <button 
                key={link.href}
                onClick={() => scrollTo(link.href)}
                className="py-2 text-left hover:text-white/70"
              >
                {link.label}
              </button>
            ))}
            <div className="h-px bg-white/10 my-2" />
            <button onClick={() => scrollTo('#system')} className="btn w-full justify-center">
              Request Access
            </button>
          </div>
        )}
      </nav>

      {/* HERO — The most important section */}
      <section className="relative min-h-[100dvh] flex items-center justify-center pt-20 overflow-hidden">
        {/* Subtle technical grid */}
        <div className="industrial-grid absolute inset-0 opacity-40" />

        {/* Central 3D Floating Hardware Object */}
        <div className="absolute inset-0 z-10 flex items-center justify-center pointer-events-none">
          <div className="w-[680px] h-[680px] md:w-[820px] md:h-[820px] pointer-events-auto">
            <FloatingCore className="w-full h-full" />
          </div>
        </div>

        <div className="relative z-20 max-w-5xl px-6 text-center">
          {/* Technical label */}
          <div className="inline-flex items-center gap-2 mb-8 border border-white/10 rounded-full px-4 py-1 text-[10px] tracking-[2px] text-[#a3a3a3]">
            <div className="w-px h-3 bg-white/30" />
            PROTOCOL 04 • PRECISION DIVISION
          </div>

          {/* Main Headline — Ultra minimal, powerful */}
          <h1 className="heading-display text-[92px] md:text-[120px] leading-[0.88] tracking-[-4.8px] mb-6">
            ENGINEERED<br />FOR THE<br />NEXT ERA
          </h1>

          <p className="max-w-md mx-auto text-lg text-[#a3a3a3] tracking-[-0.2px] mb-12">
            Industrial technology built with the rigor of precision engineering<br />and the soul of human intention.
          </p>

          {/* CTA Row */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button 
              onClick={() => document.getElementById('work')?.scrollIntoView({ behavior: 'smooth' })}
              className="btn btn-primary group text-sm px-8 py-4 flex items-center gap-3"
            >
              Explore the system
              <ArrowRight className="group-hover:translate-x-0.5 transition-transform" size={16} />
            </button>
            
            <button className="btn text-sm px-7 py-4 border-white/10">
              Watch the film
            </button>
          </div>

          {/* Micro technical indicators */}
          <div className="mt-16 flex justify-center gap-8 text-[10px] font-mono tracking-[1.5px] text-[#666]">
            <div>PRECISION · 0.008mm</div>
            <div>MATERIAL · TITANIUM + GLASS</div>
            <div>ORIGIN · 2032</div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-30">
          <div className="text-[10px] tracking-[2px] text-[#666]">SCROLL TO BEGIN</div>
          <motion.div 
            animate={{ y: [0, 6, 0] }} 
            transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
            className="w-px h-8 bg-gradient-to-b from-white/40 to-transparent"
          />
        </div>
      </section>

      {/* SYSTEM */}
      <System />

      {/* PROCESS */}
      <Process />

      {/* SHOWCASE */}
      <Showcase />

      {/* EXPERIENCE */}
      <Experience />

      {/* OBJECTS */}
      <Objects />

      {/* CLOSING + FOOTER */}
      <Closing />
    </div>
  );
}
