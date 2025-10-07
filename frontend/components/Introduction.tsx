import React from 'react';
import { ParticleCard, useMobileDetection } from './MagicBento';

export const Introduction: React.FC = () => {
 const isMobile = useMobileDetection();
 const shouldDisableAnimations = isMobile;

 return (
 <div className="mb-8 px-6 md:px-8">
 <ParticleCard
 disableAnimations={shouldDisableAnimations}
 className="card card--border-glow"
 enableTilt={true}
 clickEffect={true}
 enableMagnetism={true}
 >
 <div className="text-center p-4">
 <h2 className="text-2xl md:text-3xl font-bold text-white mb-3">Welcome to the Exoplanet Classifier</h2>
 <p className="text-slate-400 max-w-3xl mx-auto">
 This tool leverages AI to analyze astronomical data from Kepler Objects of Interest (KOIs).
 Input the parameters of a celestial object below to determine if it's a confirmed exoplanet, a candidate, or a false positive.
 </p>
 </div>
 </ParticleCard>
 </div>
 );
};
