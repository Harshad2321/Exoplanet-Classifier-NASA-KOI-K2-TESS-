import React from 'react';
import type { ExoplanetData } from '../types';

interface SliderInputProps {
  id: keyof ExoplanetData;
  value: number;
  onChange: (id: keyof ExoplanetData, value: number) => void;
}

export const SliderInput: React.FC<SliderInputProps> = ({ id, value, onChange }) => {
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(id, parseFloat(e.target.value));
  };
  
  const backgroundStyle = {
    background: `linear-gradient(to right, #4f46e5 ${value * 100}%, #374151 ${value * 100}%)`
  };

  return (
    <div className="w-full px-4">
      <div className="text-center mb-2">
        <span className="font-mono text-lg text-white">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        id={id}
        name={id}
        min="0"
        max="1"
        step="0.01"
        value={value}
        onChange={handleSliderChange}
        className="w-full h-2 rounded-lg appearance-none cursor-pointer focus:outline-none"
        style={backgroundStyle}
      />
    </div>
  );
};