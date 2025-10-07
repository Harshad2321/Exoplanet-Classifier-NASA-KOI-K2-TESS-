import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { NumericInput } from './components/NumericInput';
import { SliderInput } from './components/SliderInput';
import { ClassificationResult } from './components/ClassificationResult';
import { BatchResults } from './components/BatchResults';
import Particles from './components/Particles';
import { INPUT_FIELDS } from './constants';

import { Classification, type ExoplanetData, type ClassificationResponse } from './types';
import { BentoCardGrid, ParticleCard, useMobileDetection } from './components/MagicBento';
import './components/MagicBento.css';
import { Tooltip } from './components/Tooltip';
import { Introduction } from './components/Introduction';
import { FileInput } from './components/FileInput';

const initialData = INPUT_FIELDS.reduce((acc, field) => {
 acc[field.id as keyof ExoplanetData] = field.defaultValue;
 return acc;
}, { dispositionScore: 0.57 } as Partial<ExoplanetData>) as ExoplanetData;

const loadingMessages = [
 "Analyzing light curves...",
 "Calculating orbital data...",
 "Querying AI model...",
 "Finalizing classification...",
];

const App: React.FC = () => {
 const [formData, setFormData] = useState<ExoplanetData>(initialData);
 const [isLoading, setIsLoading] = useState<boolean>(false);
 const [result, setResult] = useState<ClassificationResponse | null>(null);
 const [error, setError] = useState<string | null>(null);
 const [fileError, setFileError] = useState<string | null>(null);
 const [loadingMessage, setLoadingMessage] = useState<string>(loadingMessages[0]);
 const [messageOpacity, setMessageOpacity] = useState(1);
 const [batchResults, setBatchResults] = useState<any>(null);
 const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);

 const gridRef = useRef(null);
 const isMobile = useMobileDetection();
 const shouldDisableAnimations = isMobile;

 useEffect(() => {
 let intervalId: number;
 let timeoutId: number;

 if (isLoading) {
 let currentIndex = 0;
 setLoadingMessage(loadingMessages[0]);
 setMessageOpacity(1);

 intervalId = window.setInterval(() => {
 setMessageOpacity(0);

 timeoutId = window.setTimeout(() => {
 currentIndex = (currentIndex + 1) % loadingMessages.length;
 setLoadingMessage(loadingMessages[currentIndex]);
 setMessageOpacity(1);
 }, 300);

 }, 1500);
 }

 return () => {
 if (intervalId) window.clearInterval(intervalId);
 if (timeoutId) window.clearTimeout(timeoutId);
 setLoadingMessage(loadingMessages[0]);
 setMessageOpacity(1);
 };
 }, [isLoading]);

 const handleInputChange = useCallback((id: keyof ExoplanetData, value: number) => {
 setFormData(prev => ({ ...prev, [id]: value }));
 }, []);

 const handleFileParsed = useCallback((data: Partial<ExoplanetData>) => {

 const sanitizedData = Object.entries(data).reduce((acc, [key, value]) => {
 if (key in initialData && typeof value === 'number' && isFinite(value)) {
 acc[key as keyof ExoplanetData] = value;
 }
 return acc;
 }, {} as Partial<ExoplanetData>);

 setFormData(prev => ({ ...prev, ...sanitizedData }));
 setFileError(null);
 }, []);

 const handleParseError = useCallback((message: string) => {
 setFileError(message);
 }, []);

 const handleBatchUpload = useCallback(async (file: File) => {
 setIsLoading(true);
 setBatchResults(null);
 setResult(null);
 setError(null);
 setFileError(null);
 setUploadedFileName(file.name);

 try {
 const formData = new FormData();
 formData.append('file', file);

 const response = await fetch('/api/classify-batch', {
 method: 'POST',
 body: formData,
 });

 if (!response.ok) {
 throw new Error(`HTTP error! status: ${response.status}`);
 }

 const data = await response.json();

 if (data.total_rows === 1 && data.results.length === 1) {

 const singleResult = data.results[0];
 const confidence_pct = (singleResult.confidence * 100).toFixed(1);
 let rationale = '';

 if (singleResult.classification === 'CONFIRMED') {
 rationale = `High confidence (${confidence_pct}%) confirmed exoplanet based on NASA AI analysis of orbital and stellar parameters.`;
 } else if (singleResult.classification === 'CANDIDATE') {
 rationale = `Moderate confidence (${confidence_pct}%) exoplanet candidate requiring further observation and verification.`;
 } else {
 rationale = `High confidence (${confidence_pct}%) this is likely a false positive, not a true exoplanet detection.`;
 }

 setResult({
 classification: singleResult.classification,
 rationale: rationale,
 confidence: singleResult.confidence,
 probabilities: singleResult.probabilities
 });
 } else {

 setBatchResults(data);
 }
 } catch (err) {
 console.error("Error in batch classification:", err);
 setFileError(err instanceof Error ? err.message : "Failed to process file");
 } finally {
 setIsLoading(false);
 }
 }, []);

 const handleDownloadResults = useCallback(() => {
 if (!batchResults) return;

 const headers = ['Row', 'Classification', 'Confidence', 'CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'];
 const rows = batchResults.results.map((r: any) => [
 r.row_number,
 r.classification,
 (r.confidence * 100).toFixed(2) + '%',
 (r.probabilities.CONFIRMED * 100).toFixed(2) + '%',
 (r.probabilities.CANDIDATE * 100).toFixed(2) + '%',
 (r.probabilities['FALSE POSITIVE'] * 100).toFixed(2) + '%'
 ]);

 const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
 const blob = new Blob([csvContent], { type: 'text/csv' });
 const url = URL.createObjectURL(blob);
 const a = document.createElement('a');
 a.href = url;
 a.download = 'exoplanet_classification_results.csv';
 document.body.appendChild(a);
 a.click();
 document.body.removeChild(a);
 URL.revokeObjectURL(url);
 }, [batchResults]);

 const classifyExoplanet = async (data: ExoplanetData): Promise<ClassificationResponse> => {
 const response = await fetch('/api/classify', {
 method: 'POST',
 headers: {
 'Content-Type': 'application/json',
 },
 body: JSON.stringify(data),
 });

 if (!response.ok) {
 const errorData = await response.json().catch(() => ({ message: 'An unknown error occurred with the classification service.' }));
 throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
 }

 const classificationResult = await response.json() as ClassificationResponse;

 if (!classificationResult.classification || !Object.values(Classification).includes(classificationResult.classification)) {
 throw new Error(`Invalid classification received from the server: ${classificationResult.classification}`);
 }

 return classificationResult;
 };

 const handleSubmit = async (e: React.FormEvent) => {
 e.preventDefault();
 setIsLoading(true);
 setResult(null);
 setError(null);

 if (JSON.stringify(formData) === JSON.stringify(initialData)) {
 setTimeout(() => {
 setResult({
 classification: Classification.FALSE_POSITIVE,
 rationale: "Hold up. Our systems are showing this star is a 'Sun' and this planet is a... wait for it... 'Earth.' This isn't a new discovery, this is your home address. Please reset and try again, we're looking for exotic exoplanets, not the one outside your window. ‍"
 });
 setIsLoading(false);
 }, 1500);
 return;
 }

 try {
 const classificationResult = await classifyExoplanet(formData);
 setResult(classificationResult);
 } catch (err) {
 console.error("Error classifying exoplanet:", err);
 setError(err instanceof Error ? err.message : "An unknown error occurred.");
 } finally {
 setIsLoading(false);
 }
 };

 const coordinateFieldIds = ['rightAscension', 'declination'];
 const regularInputFields = INPUT_FIELDS.filter(field => !coordinateFieldIds.includes(field.id));
 const rightAscensionField = INPUT_FIELDS.find(f => f.id === 'rightAscension')!;
 const declinationField = INPUT_FIELDS.find(f => f.id === 'declination')!;

 return (
 <div className="min-h-screen text-slate-300 flex flex-col items-center p-4 pt-6 font-sans relative overflow-hidden">
 <div className="absolute inset-0 z-0 pointer-events-none">
 <Particles
 particleCount={1800}
 particleSpread={10}
 speed={0.05}
 particleBaseSize={110.4}
 alphaParticles={true}
 cameraDistance={25}
 moveParticlesOnHover={true}
 particleHoverFactor={0.5}
 />
 </div>
 <Header />
 <main className="w-full max-w-6xl mx-auto flex-grow mb-10 z-10">
 <Introduction />
 <FileInput
 onFileParsed={handleFileParsed}
 onParseError={handleParseError}
 onBatchUpload={handleBatchUpload}
 />
 {fileError && <p className="text-red-400 text-center mt-2 text-sm">{fileError}</p>}
 {uploadedFileName && (
 <p className="text-center text-indigo-400 text-sm mt-2">
 Uploaded: {uploadedFileName}
 </p>
 )}

 {!batchResults && (
 <div className="p-6 md:p-8">
 <h3 className="text-xl font-bold text-center mb-6 text-white/90">Or Enter Data Manually</h3>
 <form onSubmit={handleSubmit}>
 <BentoCardGrid gridRef={gridRef} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-6">
 {regularInputFields.map(field => (
 <ParticleCard
 key={field.id}
 disableAnimations={shouldDisableAnimations}
 className="card card--border-glow"
 enableTilt={true}
 clickEffect={true}
 enableMagnetism={true}
 >
 <div className="text-center mb-2">
 <label htmlFor={field.id} className="flex items-center justify-center text-sm font-medium text-slate-300">
 {field.label} {field.unit && `(${field.unit})`}
 <span className="ml-2">
 <Tooltip text={field.tooltip} />
 </span>
 </label>
 </div>
 <div className="flex-grow flex items-center justify-center">
 <NumericInput
 id={field.id as keyof ExoplanetData}
 label={field.label}
 value={formData[field.id as keyof ExoplanetData]}
 onChange={handleInputChange}
 step={field.step}
 precision={field.precision}
 />
 </div>
 </ParticleCard>
 ))}
 {}
 <ParticleCard
 key="coordinates"
 disableAnimations={shouldDisableAnimations}
 className="card card--border-glow md:col-span-2 lg:col-span-3"
 enableTilt={true}
 clickEffect={true}
 enableMagnetism={true}
 >
 <div className="flex flex-col md:flex-row items-stretch justify-around w-full h-full gap-x-8 gap-y-4">
 {}
 <div className="flex-1 flex flex-col items-center">
 <div className="text-center mb-2">
 <label htmlFor={rightAscensionField.id} className="flex items-center justify-center text-sm font-medium text-slate-300">
 {rightAscensionField.label} {rightAscensionField.unit && `(${rightAscensionField.unit})`}
 <span className="ml-2">
 <Tooltip text={rightAscensionField.tooltip} />
 </span>
 </label>
 </div>
 <div className="flex-grow flex items-center justify-center w-full">
 <NumericInput
 id={rightAscensionField.id as keyof ExoplanetData}
 label={rightAscensionField.label}
 value={formData[rightAscensionField.id as keyof ExoplanetData]}
 onChange={handleInputChange}
 step={rightAscensionField.step}
 precision={rightAscensionField.precision}
 />
 </div>
 </div>

 {}
 <div className="hidden md:block w-px bg-slate-700"></div>

 {}
 <div className="flex-1 flex flex-col items-center">
 <div className="text-center mb-2">
 <label htmlFor={declinationField.id} className="flex items-center justify-center text-sm font-medium text-slate-300">
 {declinationField.label} {declinationField.unit && `(${declinationField.unit})`}
 <span className="ml-2">
 <Tooltip text={declinationField.tooltip} />
 </span>
 </label>
 </div>
 <div className="flex-grow flex items-center justify-center w-full">
 <NumericInput
 id={declinationField.id as keyof ExoplanetData}
 label={declinationField.label}
 value={formData[declinationField.id as keyof ExoplanetData]}
 onChange={handleInputChange}
 step={declinationField.step}
 precision={declinationField.precision}
 />
 </div>
 </div>
 </div>
 </ParticleCard>
 <ParticleCard
 disableAnimations={shouldDisableAnimations}
 className="card card--border-glow md:col-span-2 lg:col-span-3"
 enableTilt={true}
 clickEffect={true}
 enableMagnetism={true}
 >
 <div className="text-center mb-2">
 <label htmlFor="dispositionScore" className="flex items-center justify-center text-sm font-medium text-slate-300">
 KOI Disposition Score
 <span className="ml-2">
 <Tooltip text="A score from 0 to 1 indicating the confidence in the KOI being a planetary candidate." />
 </span>
 </label>
 </div>
 <div className="flex-grow flex items-center justify-center w-full">
 <SliderInput
 id="dispositionScore"
 value={formData.dispositionScore}
 onChange={handleInputChange}
 />
 </div>
 </ParticleCard>
 </BentoCardGrid>

 <div className="mt-10 flex justify-center">
 <button
 type="submit"
 disabled={isLoading}
 className={`group relative flex items-center justify-center w-full max-w-xs px-8 py-3 bg-indigo-600 text-white font-bold rounded-lg shadow-lg hover:bg-indigo-500 disabled:bg-slate-600 disabled:cursor-not-allowed transition-all duration-300 ease-in-out transform hover:scale-105 overflow-hidden ${isLoading ? 'animate-bg-scan animate-pulse-glow' : ''}`}
 >
 {isLoading ? (
 <>
 <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
 <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
 <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
 </svg>
 <span style={{ opacity: messageOpacity }} className="transition-opacity duration-300">{loadingMessage}</span>
 </>
 ) : (
 <>
 <svg className="w-6 h-6 mr-3 transform -rotate-45 group-hover:rotate-0 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
 </svg>
 Classify Exoplanet
 </>
 )}
 </button>
 </div>
 </form>

 {}
 {(result || error) && (
 <div className="mt-8 pt-6 border-t border-slate-700">
 <ClassificationResult result={result} error={error} />
 </div>
 )}
 </div>
 )}

 {}
 {batchResults && (
 <div className="mt-8 pt-6 border-t border-slate-700 p-6 md:p-8">
 <h3 className="text-2xl font-bold text-center mb-4 text-white">
 Classification Results ({batchResults.total_rows} {batchResults.total_rows === 1 ? 'Planet' : 'Planets'})
 </h3>
 <BatchResults
 results={batchResults.results || []}
 successful={batchResults.successful || 0}
 failed={batchResults.failed || 0}
 total={batchResults.total_rows || 0}
 onDownload={handleDownloadResults}
 />
 </div>
 )}
 </main>
 <Footer />
 </div>
 );
};

export default App;