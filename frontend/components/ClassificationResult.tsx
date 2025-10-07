
import React from 'react';
import { Classification, ClassificationResponse } from '../types';

interface ClassificationResultProps {
 result: ClassificationResponse | null;
 error: string | null;
}

const resultStyles = {
 [Classification.CONFIRMED]: {
 label: "Confirmed Exoplanet",
 bgColor: "bg-green-500/10",
 textColor: "text-green-400",
 borderColor: "border-green-500/30",
 icon: (
 <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
 )
 },
 [Classification.CANDIDATE]: {
 label: "Planetary Candidate",
 bgColor: "bg-yellow-500/10",
 textColor: "text-yellow-400",
 borderColor: "border-yellow-500/30",
 icon: (
 <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
 )
 },
 [Classification.FALSE_POSITIVE]: {
 label: "False Positive",
 bgColor: "bg-red-500/10",
 textColor: "text-red-400",
 borderColor: "border-red-500/30",
 icon: (
 <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
 )
 },
};

export const ClassificationResult: React.FC<ClassificationResultProps> = ({ result, error }) => {
 if (error) {
 return (
 <div className={`p-4 rounded-lg border bg-red-900/20 border-red-500/30 text-red-400`}>
 <h3 className="font-bold mb-2">Error</h3>
 <p>{error}</p>
 </div>
 );
 }

 if (!result) {
 return null;
 }

 const { classification, rationale } = result;
 const style = resultStyles[classification];

 return (
 <div className="animate-fade-in">
 <h2 className="text-2xl font-bold text-center mb-4 text-white">Classification Result</h2>
 <div className={`p-6 rounded-xl border ${style.bgColor} ${style.borderColor}`}>
 <div className="flex flex-col md:flex-row items-center gap-4">
 <div className={`flex-shrink-0 ${style.textColor}`}>
 {style.icon}
 </div>
 <div className="text-center md:text-left">
 <p className={`text-xl font-bold ${style.textColor}`}>{style.label}</p>
 <p className="text-slate-300 mt-1">{rationale}</p>
 </div>
 </div>
 </div>
 </div>
 );
};
