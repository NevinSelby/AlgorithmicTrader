'use client'

import { useState } from 'react'

interface MLPredictorProps {
  symbol: string
}

export default function MLPredictor({ symbol }: MLPredictorProps) {
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState<any>(null)

  const generatePrediction = async () => {
    setLoading(true)
    try {
      const response = await fetch(`/api/ml/predict/${symbol}`)
      if (response.ok) {
        const data = await response.json()
        setPrediction(data)
      }
    } catch (error) {
      console.error('Error generating prediction:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 backdrop-blur-xl rounded-2xl shadow-2xl p-6 animate-slide-up border border-slate-700/50">
      <h2 className="text-2xl font-bold mb-4 text-slate-200">AI Price Prediction</h2>
      
      <p className="text-slate-400 mb-6">
        Use our advanced ML model to predict future stock prices based on historical patterns and technical indicators.
      </p>
      
      <button
        onClick={generatePrediction}
        disabled={loading}
        className="px-6 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-xl hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold shadow-lg hover:shadow-xl hover:scale-105"
      >
        {loading ? 'Generating Prediction...' : 'Generate Prediction'}
      </button>
      
      {prediction && (
        <div className="mt-6 p-6 bg-slate-800/80 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-700/50">
          <h3 className="text-lg font-semibold mb-4 text-slate-200">Prediction Results</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-blue-600/20 p-4 rounded-xl border border-blue-600/30">
              <p className="text-sm text-slate-400">Current Price</p>
              <p className="text-2xl font-bold text-blue-400">${prediction.currentPrice.toFixed(2)}</p>
            </div>
            <div className="bg-purple-600/20 p-4 rounded-xl border border-purple-600/30">
              <p className="text-sm text-slate-400">Predicted (7d)</p>
              <p className="text-2xl font-bold text-purple-400">${prediction.predictedPrice7d.toFixed(2)}</p>
            </div>
            <div className="bg-indigo-600/20 p-4 rounded-xl border border-indigo-600/30">
              <p className="text-sm text-slate-400">Predicted (30d)</p>
              <p className="text-2xl font-bold text-indigo-400">${prediction.predictedPrice30d.toFixed(2)}</p>
            </div>
            <div className="bg-amber-600/20 p-4 rounded-xl border border-amber-600/30">
              <p className="text-sm text-slate-400">Confidence</p>
              <p className="text-2xl font-bold text-amber-400">{prediction.confidence.toFixed(1)}%</p>
            </div>
          </div>
          
          {prediction.trend && (
            <div className="mt-4 bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
              <p className="text-sm text-slate-400 mb-2">AI Prediction Trend:</p>
              <p className={`text-2xl font-bold ${prediction.trend === 'bullish' ? 'text-emerald-400' : prediction.trend === 'bearish' ? 'text-red-400' : 'text-slate-400'}`}>
                {prediction.trend.toUpperCase()}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

