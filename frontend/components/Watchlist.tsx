'use client'

import { useState, useEffect } from 'react'

interface StockData {
  symbol: string
  price: number
  change: number
  changePercent: number
}

interface WatchlistProps {
  onSelectStock?: (symbol: string) => void
}

export default function Watchlist({ onSelectStock }: WatchlistProps = {}) {
  const [watchlist, setWatchlist] = useState<string[]>([])
  const [stocks, setStocks] = useState<StockData[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    // Load watchlist from localStorage
    const saved = localStorage.getItem('iterai-watchlist')
    if (saved) {
      const parsed = JSON.parse(saved)
      setWatchlist(parsed)
      if (parsed.length > 0) {
        fetchStocks(parsed)
      }
    }
  }, [])

  const fetchStocks = async (symbols: string[]) => {
    setLoading(true)
    try {
      const promises = symbols.map(symbol => 
        fetch(`/api/stock/${symbol}`).then(res => res.ok ? res.json() : null)
      )
      const results = await Promise.all(promises)
      const validStocks = results.filter(Boolean)
      setStocks(validStocks)
    } catch (error) {
      console.error('Error fetching stocks:', error)
    } finally {
      setLoading(false)
    }
  }

  const addToWatchlist = (symbol: string) => {
    if (!symbol || watchlist.includes(symbol)) return
    const updated = [...watchlist, symbol]
    setWatchlist(updated)
    localStorage.setItem('iterai-watchlist', JSON.stringify(updated))
    fetchStocks(updated)
  }

  const removeFromWatchlist = (symbol: string) => {
    const updated = watchlist.filter(s => s !== symbol)
    setWatchlist(updated)
    localStorage.setItem('iterai-watchlist', JSON.stringify(updated))
    setStocks(stocks.filter(s => s.symbol !== symbol))
  }

  const refreshWatchlist = () => {
    if (watchlist.length > 0) {
      fetchStocks(watchlist)
    }
  }

  return (
    <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg p-6 mb-6 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-slate-200">Watchlist</h2>
        <button
          onClick={refreshWatchlist}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all text-sm"
        >
          🔄 Refresh
        </button>
      </div>

      {watchlist.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-slate-400 mb-4">No stocks in watchlist</p>
          <p className="text-xs text-slate-500">Search for stocks and add them to your watchlist using the ⭐ button</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {stocks.map((stock, idx) => (
            <div
              key={idx}
              onClick={() => onSelectStock && onSelectStock(stock.symbol)}
              className="bg-slate-900/60 rounded-lg p-4 border border-slate-700/30 hover:border-slate-700/50 transition-all cursor-pointer group"
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-lg font-bold text-white group-hover:text-blue-400 transition-colors">{stock.symbol}</p>
                  <p className={`text-2xl font-bold ${stock.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    ${stock.price.toFixed(2)}
                  </p>
                  <p className={`text-sm ${stock.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFromWatchlist(stock.symbol)
                  }}
                  className="text-slate-400 hover:text-red-400 transition-colors"
                >
                  ✕
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

