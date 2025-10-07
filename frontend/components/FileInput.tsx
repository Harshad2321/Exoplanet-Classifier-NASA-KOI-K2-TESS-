import React, { useState, useCallback, useRef } from 'react';
import { ParticleCard, useMobileDetection } from './MagicBento';
import type { ExoplanetData } from '../types';

interface FileInputProps {
 onFileParsed: (data: Partial<ExoplanetData>) => void;
 onParseError: (message: string) => void;
 onBatchUpload?: (file: File) => void;
 enableBatchMode?: boolean;
}

const UploadIcon = () => (
 <svg className="w-8 h-8 text-slate-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-4-4V7a4 4 0 014-4h5l2 3h9a2 2 0 012 2v7a2 2 0 01-2 2H9a2 2 0 00-2 2v1a2 2 0 01-2 2H7z"></path></svg>
);

const FileIcon = () => (
 <svg className="w-8 h-8 text-indigo-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
);

export const FileInput: React.FC<FileInputProps> = ({
 onFileParsed,
 onParseError,
 onBatchUpload,
 enableBatchMode = false
}) => {
 const [isDragActive, setIsDragActive] = useState(false);
 const [fileName, setFileName] = useState<string | null>(null);
 const inputRef = useRef<HTMLInputElement>(null);

 const isMobile = useMobileDetection();
 const shouldDisableAnimations = isMobile;

 const parseCSV = (text: string): Partial<ExoplanetData> => {
 const lines = text.trim().split(/\r?\n/);
 if (lines.length < 2) throw new Error("CSV file must have a header and at least one data row.");

 const headers = lines[0].split(',').map(h => h.trim());
 const values = lines[1].split(',').map(v => v.trim());

 const data = headers.reduce((obj, header, index) => {
 const value = parseFloat(values[index]);
 if (!isNaN(value)) {
 obj[header] = value;
 }
 return obj;
 }, {} as Record<string, number>);

 return data;
 }

 const processFile = useCallback((file: File) => {
 onParseError('');

 if (file.name.endsWith('.csv') && onBatchUpload) {
 setFileName(file.name);
 onBatchUpload(file);
 return;
 }

 const reader = new FileReader();

 reader.onload = (event) => {
 try {
 const content = event.target?.result as string;
 let data: Partial<ExoplanetData>;

 if (file.type === 'application/json' || file.name.endsWith('.json')) {
 data = JSON.parse(content);
 } else if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
 data = parseCSV(content);
 } else {
 throw new Error("Unsupported file type. Please upload a JSON or CSV file.");
 }

 if (typeof data !== 'object' || data === null) {
 throw new Error("Invalid file content. Must be a JSON object or a valid CSV.");
 }

 onFileParsed(data);
 setFileName(file.name);
 } catch (e) {
 const message = e instanceof Error ? e.message : "An unknown error occurred during file parsing.";
 onParseError(message);
 setFileName(null);
 }
 };

 reader.onerror = () => {
 onParseError("Failed to read the file.");
 setFileName(null);
 };

 reader.readAsText(file);
 }, [onFileParsed, onParseError, onBatchUpload, enableBatchMode]);

 const handleDrag = useCallback((e: React.DragEvent) => {
 e.preventDefault();
 e.stopPropagation();
 if (e.type === "dragenter" || e.type === "dragover") {
 setIsDragActive(true);
 } else if (e.type === "dragleave") {
 setIsDragActive(false);
 }
 }, []);

 const handleDrop = useCallback((e: React.DragEvent) => {
 e.preventDefault();
 e.stopPropagation();
 setIsDragActive(false);
 if (e.dataTransfer.files && e.dataTransfer.files[0]) {
 processFile(e.dataTransfer.files[0]);
 }
 }, [processFile]);

 const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
 e.preventDefault();
 if (e.target.files && e.target.files[0]) {
 processFile(e.target.files[0]);
 }
 };

 const handleClearFile = () => {
 setFileName(null);
 onParseError('');
 if (inputRef.current) {
 inputRef.current.value = "";
 }
 };

 const handleClick = () => {
 inputRef.current?.click();
 };

 return (
 <div className="mb-8 px-6 md:px-8">
 <ParticleCard
 disableAnimations={shouldDisableAnimations}
 className="card card--border-glow"
 enableTilt={true}
 clickEffect={false}
 enableMagnetism={true}
 >
 <div
 className={`relative p-4 text-center transition-colors duration-300 ${isDragActive ? 'bg-white/5' : ''}`}
 onDragEnter={handleDrag}
 onDragOver={handleDrag}
 onDragLeave={handleDrag}
 onDrop={handleDrop}
 >
 <input
 ref={inputRef}
 type="file"
 id="file-upload"
 className="hidden"
 accept=".json,.csv"
 onChange={handleChange}
 />

 {!fileName ? (
 <div onClick={handleClick} className="cursor-pointer border-2 border-dashed border-slate-600 hover:border-slate-500 rounded-lg p-8 flex flex-col items-center justify-center">
 <UploadIcon />
 <p className="text-slate-300 font-semibold">Upload Data File</p>
 <p className="text-sm text-slate-400 mt-1">Drag & drop or click to upload</p>
 <p className="text-xs text-slate-500 mt-2">(Supports .JSON or .CSV)</p>
 </div>
 ) : (
 <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 flex flex-col items-center justify-center">
 <div className="flex items-center">
 <FileIcon />
 <span className="text-white font-medium">{fileName}</span>
 </div>
 <button
 onClick={handleClearFile}
 className="mt-4 text-sm bg-red-500/20 hover:bg-red-500/40 text-red-400 px-3 py-1 rounded-md transition-colors"
 >
 Clear
 </button>
 </div>
 )}
 </div>
 </ParticleCard>
 </div>
 );
};
