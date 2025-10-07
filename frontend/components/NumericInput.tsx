import React, { useState, useEffect, useRef } from 'react';
import type { ExoplanetData } from '../types';

interface NumericInputProps {
 id: keyof ExoplanetData;
 label: string;
 value: number;
 onChange: (id: keyof ExoplanetData, value: number) => void;
 step: number;
 precision: number;
}

const formatNumberForDisplay = (num: number, precision: number): string => {
 if (num === 0) return num.toFixed(precision);
 const absNum = Math.abs(num);

 if (absNum > 1e7 || (absNum < 1e-4 && absNum > 0)) {
 return num.toExponential(precision);
 }
 return num.toFixed(precision);
};

export const NumericInput: React.FC<NumericInputProps> = ({ id, label, value, onChange, step, precision }) => {
 const [displayValue, setDisplayValue] = useState(formatNumberForDisplay(value, precision));
 const isFocused = useRef(false);

 useEffect(() => {
 if (!isFocused.current) {
 setDisplayValue(formatNumberForDisplay(value, precision));
 }
 }, [value, precision]);

 const commitChange = (val: string) => {
 const parsedValue = parseFloat(val);
 if (!isNaN(parsedValue) && isFinite(parsedValue)) {
 if (parsedValue !== value) {
 onChange(id, parsedValue);
 } else {

 setDisplayValue(formatNumberForDisplay(value, precision));
 }
 } else {

 setDisplayValue(formatNumberForDisplay(value, precision));
 }
 };

 const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
 setDisplayValue(e.target.value);
 };

 const handleBlur = () => {
 isFocused.current = false;
 commitChange(displayValue);
 };

 const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
 isFocused.current = true;
 e.target.select();
 };

 const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
 if (e.key === 'Enter') {
 commitChange(displayValue);
 e.currentTarget.blur();
 } else if (e.key === 'Escape') {
 setDisplayValue(formatNumberForDisplay(value, precision));
 e.currentTarget.blur();
 }
 };

 const handleStep = (direction: 'up' | 'down') => {
 const currentValue = parseFloat(displayValue);
 const baseValue = !isNaN(currentValue) ? currentValue : value;
 const nextValue = baseValue + (direction === 'up' ? step : -step);

 if (isFinite(nextValue)) {
 onChange(id, nextValue);
 }
 };

 return (
 <div className="w-full">
 <div className="flex items-center transition-all duration-200">
 <button
 type="button"
 onClick={() => handleStep('down')}
 className="px-2.5 py-1.5 text-slate-400 hover:text-white hover:bg-white/10 rounded-l-lg transition"
 aria-label={`Decrement ${label}`}
 >
 -
 </button>
 <input
 type="text"
 inputMode="decimal"
 id={id}
 name={id}
 value={displayValue}
 onChange={handleInputChange}
 onBlur={handleBlur}
 onFocus={handleFocus}
 onKeyDown={handleKeyDown}
 className="w-full bg-transparent text-center text-white font-mono text-lg py-1.5 outline-none border-x border-slate-700"
 />
 <button
 type="button"
 onClick={() => handleStep('up')}
 className="px-2.5 py-1.5 text-slate-400 hover:text-white hover:bg-white/10 rounded-r-lg transition"
 aria-label={`Increment ${label}`}
 >
 +
 </button>
 </div>
 </div>
 );
};
