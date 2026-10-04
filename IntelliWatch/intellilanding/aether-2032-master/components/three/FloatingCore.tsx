'use client';

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface FloatingCoreProps {
  className?: string;
}

/**
 * AETHER 2032 — Central Precision Core
 * 
 * Highly refined floating industrial hardware object.
 * Multi-layered transparent glass with internal mechanical architecture.
 * Magnetic mouse interaction with layered parallax.
 * Laboratory-grade lighting + subtle functional red indicators.
 */

function PrecisionCore({ mouse }: { mouse: React.MutableRefObject<{ x: number; y: number }> }) {
  const mainGroup = useRef<THREE.Group>(null!);
  const outerShell = useRef<THREE.Mesh>(null!);
  const middleShell = useRef<THREE.Mesh>(null!);
  const innerFrame = useRef<THREE.Group>(null!);
  const centralCore = useRef<THREE.Group>(null!);
  const particlesNear = useRef<THREE.Points>(null!);
  const particlesFar = useRef<THREE.Points>(null!);
  const redRing = useRef<THREE.Mesh>(null!);

  // Layered mouse response (outer glass reacts less than inner structures for depth)
  useFrame((state) => {
    if (!mainGroup.current) return;

    const t = state.clock.getElapsedTime();
    const mx = mouse.current.x;
    const my = mouse.current.y;

    // Master slow, premium rotation
    const baseRotY = t * 0.065;
    const baseRotX = Math.sin(t * 0.04) * 0.06;

    // Layered tilt — stronger on inner elements for beautiful parallax
    const outerTiltX = my * 0.22;
    const outerTiltY = mx * 0.28;

    const innerTiltX = my * 0.48;
    const innerTiltY = mx * 0.55;

    // Main group (outer shell influence)
    mainGroup.current.rotation.y = baseRotY + outerTiltY;
    mainGroup.current.rotation.x = baseRotX + outerTiltX;

    // Inner mechanical frame — stronger reaction
    if (innerFrame.current) {
      innerFrame.current.rotation.y = baseRotY * 1.1 + innerTiltY * 1.15;
      innerFrame.current.rotation.x = baseRotX * 0.9 + innerTiltX * 1.1;
    }

    // Central floating core — most reactive
    if (centralCore.current) {
      centralCore.current.rotation.y = baseRotY * 1.35 + innerTiltY * 1.4;
      centralCore.current.rotation.x = baseRotX * 1.2 + innerTiltX * 1.35;
    }

    // Very subtle breathing on outer glass (physical pressure feel)
    if (outerShell.current) {
      const breath = 1 + Math.sin(t * 0.55) * 0.0065;
      outerShell.current.scale.setScalar(breath);
    }

    // Slow, elegant particle motion
    if (particlesNear.current) {
      particlesNear.current.rotation.y = t * 0.022;
    }
    if (particlesFar.current) {
      particlesFar.current.rotation.y = t * -0.014;
    }

    // Subtle pulsing on the red calibration ring (functional indicator)
    if (redRing.current) {
      const pulse = 1 + Math.sin(t * 1.8) * 0.015;
      redRing.current.scale.setScalar(pulse);
    }

    // Very subtle mouse-reactive internal emissive glow (makes the core feel "alive")
    if (centralCore.current) {
      const glow = 0.12 + (Math.abs(mx) + Math.abs(my)) * 0.18;
      centralCore.current.traverse((child) => {
        if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshPhongMaterial) {
          child.material.emissiveIntensity = glow;
        }
      });
    }
  });

  // High-quality particle systems
  const { nearParticles, farParticles } = useMemo(() => {
    const nearCount = 36;
    const farCount = 58;

    const nearPos = new Float32Array(nearCount * 3);
    const nearSize = new Float32Array(nearCount);

    const farPos = new Float32Array(farCount * 3);
    const farSize = new Float32Array(farCount);

    // Near particles (tighter, brighter)
    for (let i = 0; i < nearCount; i++) {
      const r = 1.65 + Math.random() * 0.55;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      nearPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      nearPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.55;
      nearPos[i * 3 + 2] = r * Math.cos(phi) * 0.65;

      nearSize[i] = 0.55 + Math.random() * 0.9;
    }

    // Far particles (more dispersed, softer)
    for (let i = 0; i < farCount; i++) {
      const r = 2.45 + Math.random() * 1.1;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);

      farPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      farPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.4;
      farPos[i * 3 + 2] = r * Math.cos(phi) * 0.5;

      farSize[i] = 0.35 + Math.random() * 0.55;
    }

    return {
      nearParticles: { positions: nearPos, sizes: nearSize },
      farParticles: { positions: farPos, sizes: farSize },
    };
  }, []);

  return (
    <group ref={mainGroup}>
      {/* === LAYER 1: Outer Ultra-Clear Precision Glass Shell (highest quality) === */}
      <mesh ref={outerShell}>
        <icosahedronGeometry args={[1.94, 1]} />
        <meshPhysicalMaterial
          color="#fafafa"
          metalness={0.06}
          roughness={0.015}
          transmission={0.99}
          thickness={1.25}
          envMapIntensity={0.4}
          clearcoat={1}
          clearcoatRoughness={0.03}
          ior={1.5}
          transparent
          opacity={0.82}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* === LAYER 2: Secondary pressure shell with subtle internal tint === */}
      <mesh ref={middleShell}>
        <icosahedronGeometry args={[1.60, 0]} />
        <meshPhysicalMaterial
          color="#e2e8f0"
          metalness={0.12}
          roughness={0.05}
          transmission={0.94}
          thickness={0.95}
          ior={1.44}
          transparent
          opacity={0.6}
        />
      </mesh>

      {/* === LAYER 3: Internal Precision Mechanical Architecture (highly detailed) === */}
      <group ref={innerFrame}>
        {/* Primary equatorial precision ring - brushed metal */}
        <mesh>
          <torusGeometry args={[1.12, 0.026, 20, 72]} />
          <meshPhongMaterial
            color="#a1a1aa"
            emissive="#3f3f46"
            emissiveIntensity={0.06}
            shininess={100}
            specular="#ffffff"
          />
        </mesh>

        {/* Secondary inner support ring */}
        <mesh>
          <torusGeometry args={[0.82, 0.018, 16, 56]} />
          <meshPhongMaterial
            color="#71717a"
            shininess={90}
          />
        </mesh>

        {/* High-precision vertical + diagonal structural struts */}
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <group key={i} rotation={[0, (i * Math.PI) / 3, 0]}>
            {/* Main vertical strut */}
            <mesh>
              <cylinderGeometry args={[0.013, 0.013, 2.42, 4]} />
              <meshPhongMaterial color="#6b7280" shininess={75} />
            </mesh>
            {/* Thin diagonal reinforcement */}
            <mesh rotation={[0.7, 0, 0]}>
              <cylinderGeometry args={[0.007, 0.007, 1.6, 3]} />
              <meshPhongMaterial color="#52525b" shininess={60} />
            </mesh>
          </group>
        ))}

        {/* Floating micro calibration hex plates (very industrial) */}
        {[0, 1, 2].map((i) => (
          <mesh
            key={i}
            position={[0, (i - 1) * 0.68, 0]}
            rotation={[0.65, i * 2.1, 0.3]}
          >
            <cylinderGeometry args={[0.18, 0.18, 0.035, 6]} />
            <meshPhongMaterial color="#4b5563" shininess={95} />
          </mesh>
        ))}

        {/* Tiny connecting precision rods (adds complexity) */}
        {[0, 1, 2, 3].map((i) => (
          <mesh
            key={i}
            position={[Math.cos(i) * 0.95, (i % 2 - 0.5) * 0.4, Math.sin(i) * 0.3]}
            rotation={[i, i * 0.8, 0]}
          >
            <cylinderGeometry args={[0.005, 0.005, 0.9, 3]} />
            <meshPhongMaterial color="#3f3f46" shininess={50} />
          </mesh>
        ))}
      </group>

      {/* === LAYER 4: Central Floating Core (most reactive) === */}
      <group ref={centralCore}>
        {/* Dense central mass */}
        <mesh>
          <octahedronGeometry args={[0.48, 0]} />
          <meshPhongMaterial
            color="#374151"
            emissive="#1f2937"
            emissiveIntensity={0.15}
            shininess={110}
          />
        </mesh>

        {/* Thin equatorial precision disk */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.78, 0.78, 0.035, 48]} />
          <meshPhongMaterial
            color="#9ca3af"
            shininess={120}
          />
        </mesh>
      </group>

      {/* === Functional Red Calibration Ring (indicator light) === */}
      <mesh ref={redRing} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1.38, 0.009, 12, 72]} />
        <meshPhongMaterial
          color="#c14444"
          emissive="#9f3a3a"
          emissiveIntensity={0.65}
          shininess={140}
        />
      </mesh>

      {/* === Microscopic Particles — Near Layer === */}
      <points ref={particlesNear}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[nearParticles.positions, 3]}
          />
          <bufferAttribute
            attach="attributes-size"
            args={[nearParticles.sizes, 1]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.022}
          color="#e5e7eb"
          sizeAttenuation
          transparent
          opacity={0.75}
        />
      </points>

      {/* === Microscopic Particles — Far Layer === */}
      <points ref={particlesFar}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[farParticles.positions, 3]}
          />
          <bufferAttribute
            attach="attributes-size"
            args={[farParticles.sizes, 1]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.014}
          color="#9ca3af"
          sizeAttenuation
          transparent
          opacity={0.45}
        />
      </points>
    </group>
  );
}

function MouseInteraction({ mouse }: { mouse: React.MutableRefObject<{ x: number; y: number }> }) {
  React.useEffect(() => {
    let targetX = 0;
    let targetY = 0;

    const handleMouseMove = (e: MouseEvent) => {
      const x = ((e.clientX / window.innerWidth) - 0.5) * 2;
      const y = ((e.clientY / window.innerHeight) - 0.5) * 2;

      targetX = x * 0.72;
      targetY = -y * 0.72;
    };

    const handleMouseLeave = () => {
      targetX = 0;
      targetY = 0;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseleave', handleMouseLeave);

    // Smooth interpolation loop (more physical than direct lerp)
    const interval = setInterval(() => {
      mouse.current.x = THREE.MathUtils.lerp(mouse.current.x, targetX, 0.078);
      mouse.current.y = THREE.MathUtils.lerp(mouse.current.y, targetY, 0.078);
    }, 16);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseleave', handleMouseLeave);
      clearInterval(interval);
    };
  }, [mouse]);

  return null;
}

export default function FloatingCore({ className }: FloatingCoreProps) {
  const mouse = useRef({ x: 0, y: 0 });

  return (
    <div className={className}>
      <Canvas
        camera={{ position: [0, 0.4, 5.8], fov: 38 }}
        style={{ background: 'transparent' }}
        gl={{
          alpha: true,
          antialias: true,
          preserveDrawingBuffer: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 0.95,
        }}
      >
        {/* Laboratory-grade soft lighting */}
        <ambientLight intensity={0.28} color="#f4f4f5" />

        {/* Primary key light — clean, slightly from above-left */}
        <directionalLight
          position={[-7.5, 10, -3.5]}
          intensity={2.1}
          color="#f8f8fa"
        />

        {/* Cool rim light */}
        <directionalLight
          position={[4, -2, -9]}
          intensity={0.75}
          color="#a5b4fc"
        />

        {/* Very subtle red fill from below (functional indicator glow) */}
        <directionalLight
          position={[0, -6, 2]}
          intensity={0.35}
          color="#c14444"
        />

        <PrecisionCore mouse={mouse} />
        <MouseInteraction mouse={mouse} />

        <OrbitControls
          enablePan={false}
          enableZoom={false}
          enableRotate={false}
          enableDamping
          dampingFactor={0.12}
        />
      </Canvas>
    </div>
  );
}
