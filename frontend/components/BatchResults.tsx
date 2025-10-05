import React from 'react';

interface BatchResult {
    row_number: number;
    classification: string;
    confidence: number;
    probabilities: Record<string, number>;
    input_data: Record<string, number>;
}

interface BatchResultsProps {
    results: BatchResult[];
    successful: number;
    failed: number;
    total: number;
    onDownload?: () => void;
}

export const BatchResults: React.FC<BatchResultsProps> = ({
    results,
    successful,
    failed,
    total,
    onDownload
}) => {
    const getClassificationColor = (classification: string): string => {
        switch (classification.toUpperCase()) {
            case 'CONFIRMED':
                return 'text-green-400';
            case 'CANDIDATE':
                return 'text-yellow-400';
            case 'FALSE POSITIVE':
            case 'FALSE_POSITIVE':
                return 'text-red-400';
            default:
                return 'text-slate-400';
        }
    };

    const getClassificationEmoji = (classification: string): string => {
        switch (classification.toUpperCase()) {
            case 'CONFIRMED':
                return '✅';
            case 'CANDIDATE':
                return '🔍';
            case 'FALSE POSITIVE':
            case 'FALSE_POSITIVE':
                return '❌';
            default:
                return '❓';
        }
    };

    return (
        <div className="mt-8 space-y-4">
            {/* Summary Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                    <div className="text-2xl font-bold text-blue-400">{total}</div>
                    <div className="text-sm text-slate-400">Total Analyzed</div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                    <div className="text-2xl font-bold text-green-400">{successful}</div>
                    <div className="text-sm text-slate-400">Successful</div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                    <div className="text-2xl font-bold text-red-400">{failed}</div>
                    <div className="text-sm text-slate-400">Failed</div>
                </div>
            </div>

            {/* Download Button */}
            {onDownload && (
                <button
                    onClick={onDownload}
                    className="w-full md:w-auto px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-medium transition-colors mb-4"
                >
                    📥 Download Results as CSV
                </button>
            )}

            {/* Results Table */}
            <div className="overflow-x-auto bg-slate-800/30 rounded-lg border border-slate-700">
                <table className="w-full text-left">
                    <thead className="bg-slate-800/50">
                        <tr>
                            <th className="px-4 py-3 text-slate-300 font-semibold">#</th>
                            <th className="px-4 py-3 text-slate-300 font-semibold">Classification</th>
                            <th className="px-4 py-3 text-slate-300 font-semibold">Confidence</th>
                            <th className="px-4 py-3 text-slate-300 font-semibold">Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {results.map((result, idx) => (
                            <tr
                                key={idx}
                                className="border-t border-slate-700 hover:bg-slate-800/20 transition-colors"
                            >
                                <td className="px-4 py-3 text-slate-400">{result.row_number}</td>
                                <td className="px-4 py-3">
                                    <span className={`font-semibold ${getClassificationColor(result.classification)}`}>
                                        {getClassificationEmoji(result.classification)} {result.classification}
                                    </span>
                                </td>
                                <td className="px-4 py-3">
                                    <div className="flex items-center space-x-2">
                                        <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-indigo-500"
                                                style={{ width: `${result.confidence * 100}%` }}
                                            />
                                        </div>
                                        <span className="text-slate-300 text-sm font-mono">
                                            {(result.confidence * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </td>
                                <td className="px-4 py-3">
                                    <details className="cursor-pointer">
                                        <summary className="text-indigo-400 hover:text-indigo-300 text-sm">
                                            View Probabilities
                                        </summary>
                                        <div className="mt-2 space-y-1 text-xs">
                                            {Object.entries(result.probabilities).map(([cls, prob]) => (
                                                <div key={cls} className="flex justify-between text-slate-400">
                                                    <span>{cls}:</span>
                                                    <span className="font-mono">{(Number(prob) * 100).toFixed(1)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    </details>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Empty State */}
            {results.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                    <div className="text-4xl mb-4">🌌</div>
                    <p>No results to display</p>
                </div>
            )}
        </div>
    );
};
