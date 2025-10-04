import React from 'react';

const OrbitIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="-11.5 -10.23174 23 20.46348" className="w-8 h-8 mr-3 text-white shrink-0">
    <title>Orbit Logo</title>
    <circle cx="0" cy="0" r="2.05" fill="currentColor"/>
    <g stroke="currentColor" strokeWidth="1" fill="none">
      <ellipse rx="11" ry="4.2"/>
      <ellipse rx="11" ry="4.2" transform="rotate(60)"/>
      <ellipse rx="11" ry="4.2" transform="rotate(120)"/>
    </g>
  </svg>
);


export const Header: React.FC = () => {
  return (
    <header className="w-full max-w-4xl mx-auto mb-10 z-20">
       <div className="w-full bg-[#100D1B]/70 backdrop-blur-sm border border-slate-700/50 rounded-full flex items-center justify-between px-4 sm:px-6 py-3">
            <div className="flex items-center">
                <OrbitIcon />
                <h1 className="text-lg sm:text-xl font-bold text-white tracking-tight">Exoplanet Classifier</h1>
            </div>
            <nav className="hidden md:flex items-center space-x-6 text-sm">
                <a href="#" className="text-slate-300 hover:text-white transition-colors duration-200">Home</a>
                <a href="#contact-us" className="text-slate-300 hover:text-white transition-colors duration-200">Contact Us</a>
            </nav>
       </div>
    </header>
  );
};