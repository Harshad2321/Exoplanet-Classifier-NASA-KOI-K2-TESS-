
import React from 'react';

interface TooltipProps {
 text: string;
}

export const Tooltip: React.FC<TooltipProps> = ({ text }) => {
 return (
 <div className="group relative flex items-center">
 <svg
 className="w-4 h-4 text-slate-500 hover:text-slate-400 cursor-pointer"
 fill="none"
 stroke="currentColor"
 viewBox="0 0 24 24"
 xmlns="http://www.w3.org/2000/svg"
 >
 <path
 strokeLinecap="round"
 strokeLinejoin="round"
 strokeWidth="2"
 d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
 ></path>
 </svg>
 <div className="absolute bottom-full mb-2 w-max max-w-xs p-2 text-sm text-white bg-slate-900 border border-slate-700 rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none z-10">
 {text}
 </div>
 </div>
 );
};
