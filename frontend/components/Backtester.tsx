'use client'

import { useState } from 'react'

interface BacktestResult {
  equity: Array<{ date: string; value: number }>
  trades: number
  winRate: number
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  cagr: number
}

interface BacktesterProps {
  symbol: string
}

export default function Backtester({ symbol }: BacktesterProps) {
  const [entryCondition, setEntryCondition] = useState('RSI < 30')
  const [exitCondition, setExitCondition] = useState('RSI > 70')
  const [initialCapital, setInitialCapital] = useState(100000)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)

  const runBacktest = async () => {
    setLoading(true)
    try {
      const response = await fetch(
        `/api/backtest/${symbol}?entry_condition=${encodeURIComponent(entryCondition)}&exit_condition=${encodeURIComponent(exitCondition)}&initial_capital=${initialCapital}`
      )
      if (response.ok) {
        const data = await response.json()
        setResults(data)
      }
    } catch (error) {
      console.error('Error running backtest:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 backdrop-blur-xl rounded-2xl shadow-2xl p-6 animate-slide-up border border-slate-700/50">
      <h2 className="text-2xl font-bold mb-2 text-slate-200">Strategy Backtesting</h2>
      <p className="text-sm text-slate-400 mb-6">
        Test your trading strategies using all available indicators: RSI, SMA20, SMA50, EMA20, MACD, SIGNAL, UPPER_BB, LOWER_BB, K_PERCENT, D_PERCENT, ADX
      </p>
      
      <div className="bg-slate-900/50 p-4 rounded-xl mb-6 border border-slate-700/50">
        <p className="text-xs text-slate-400 mb-2">Examples:</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-slate-300">
          <p>• RSI {'<'} 30 (oversold buy)</p>
          <p>• RSI {'>'} 70 (overbought sell)</p>
          <p>• SMA20 {'>'} SMA50 (uptrend)</p>
          <p>• PRICE {'<'} LOWER_BB (Bollinger buy)</p>
          <p>• MACD {'>'} SIGNAL (bullish)</p>
          <p>• K_PERCENT {'<'} 20 (stochastic buy)</p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Entry Condition
          </label>
          <input
            type="text"
            value={entryCondition}
            onChange={(e) => setEntryCondition(e.target.value)}
            className="w-full px-4 py-2 rounded-xl bg-slate-900/60 text-gray-100 placeholder:text-gray-400 border border-slate-600/50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="e.g., RSI < 30"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Exit Condition
          </label>
          <input
            type="text"
            value={exitCondition}
            onChange={(e) => setExitCondition(e.target.value)}
            className="w-full px-4 py-2 rounded-xl bg-slate-900/60 text-gray-100 placeholder:text-gray-400 border border-slate-600/50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="e.g., RSI > 70"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Initial Capital ($)
          </label>
          <input
            type="number"
            value={initialCapital}
            onChange={(e) => setInitialCapital(Number(e.target.value))}
            className="w-full px-4 py-2 rounded-xl bg-slate-900/60 text-gray-100 border border-slate-600/50 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>
      
      <button
        onClick={runBacktest}
        disabled={loading}
        className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold shadow-lg hover:shadow-xl hover:scale-110 hover:-translate-y-1"
      >
        {loading ? 'Running Backtest...' : 'Run Backtest'}
      </button>
      
      {results && (
        <div className="mt-6 p-6 bg-slate-800/80 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-700/50 animate-fade-in-up">
          <h3 className="text-lg font-semibold mb-4 text-slate-200">Backtest Results</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-emerald-600/20 p-4 rounded-xl border border-emerald-600/30 hover:scale-105 transition-all">
              <p className="text-sm text-slate-400">Total Return</p>
              <p className="text-2xl font-bold text-emerald-400">{results.totalReturn.toFixed(2)}%</p>
            </div>
            <div className="bg-blue-600/20 p-4 rounded-xl border border-blue-600/30 hover:scale-105 transition-all">
              <p className="text-sm text-slate-400">Sharpe Ratio</p>
              <p className="text-2xl font-bold text-blue-400">{results.sharpeRatio.toFixed(2)}</p>
            </div>
            <div className="bg-red-600/20 p-4 rounded-xl border border-red-600/30 hover:scale-105 transition-all">
              <p className="text-sm text-slate-400">Max Drawdown</p>
              <p className="text-2xl font-bold text-red-400">{results.maxDrawdown.toFixed(2)}%</p>
            </div>
            <div className="bg-purple-600/20 p-4 rounded-xl border border-purple-600/30 hover:scale-105 transition-all">
              <p className="text-sm text-slate-400">Win Rate</p>
              <p className="text-2xl font-bold text-purple-400">{results.winRate.toFixed(2)}%</p>
            </div>
          </div>
          {results.cagr !== undefined && (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-cyan-600/20 p-4 rounded-xl border border-cyan-600/30">
                <p className="text-sm text-slate-400">Initial Capital</p>
                <p className="text-xl font-bold text-cyan-400">${results.initialCapital?.toLocaleString() || 'N/A'}</p>
              </div>
              <div className="bg-green-600/20 p-4 rounded-xl border border-green-600/30">
                <p className="text-sm text-slate-400">Final Capital</p>
                <p className="text-xl font-bold text-green-400">${results.finalCapital?.toLocaleString() || 'N/A'}</p>
              </div>
              <div className="bg-orange-600/20 p-4 rounded-xl border border-orange-600/30">
                <p className="text-sm text-slate-400">Total Trades</p>
                <p className="text-xl font-bold text-orange-400">{results.trades || 0}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

