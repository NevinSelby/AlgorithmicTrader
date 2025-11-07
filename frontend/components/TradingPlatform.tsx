'use client'

import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { Toaster } from 'react-hot-toast'
import Link from 'next/link'
import { api } from '../lib/api'

// Dynamic imports to avoid SSR issues
const PlotComponent = dynamic(() => import('./Chart'), {
  ssr: false,
})
const Backtester = dynamic(() => import('./Backtester'), {
  ssr: false,
})
const MLPredictor = dynamic(() => import('./MLPredictor'), {
  ssr: false,
})
const LandingPage = dynamic(() => import('./LandingPage'), {
  ssr: false,
})
const Watchlist = dynamic(() => import('./Watchlist'), {
  ssr: false,
})
const PopularStocks = dynamic(() => import('./PopularStocks'), {
  ssr: false,
})

interface StockData {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  timestamp: string
}

interface Indicator {
  name: string
  enabled: boolean
}

interface NewsItem {
  title: string
  link: string
  publisher: string
}

export default function TradingPlatform() {
  const [stockSymbol, setStockSymbol] = useState('AAPL')
  const [stockData, setStockData] = useState<StockData | null>(null)
  const [loading, setLoading] = useState(false)
  const [chartData, setChartData] = useState<any>(null)
  const [news, setNews] = useState<NewsItem[]>([])
  const [indicators, setIndicators] = useState<Indicator[]>([
    { name: 'SMA 20', enabled: true },
    { name: 'SMA 50', enabled: false },
    { name: 'EMA 12', enabled: false },
    { name: 'EMA 26', enabled: false },
    { name: 'RSI', enabled: true },
    { name: 'MACD', enabled: false },
    { name: 'Bollinger Bands', enabled: false },
    { name: 'Stochastic', enabled: false },
    { name: 'ADX', enabled: false },
    { name: 'Volume', enabled: true },
  ])
  const [activeTab, setActiveTab] = useState('chart')
  const [showLanding, setShowLanding] = useState(true)
  const [watchlistState, setWatchlistState] = useState(0)

  useEffect(() => {
    document.documentElement.classList.add('dark')
    fetchNews()
  }, [])

  const fetchNews = async () => {
    try {
      const response = await fetch(api.url('news'))
      if (response.ok) {
        const data = await response.json()
        const formattedNews = data.news.slice(0, 6).map((item: any) => ({
          title: item.title || 'Market News',
          link: item.link || '#',
          publisher: item.publisher || 'Financial News'
        }))
        setNews(formattedNews)
      }
    } catch (error) {
      console.error('Error fetching news:', error)
    }
  }

  const toggleIndicator = (index: number) => {
    const newIndicators = [...indicators]
    newIndicators[index].enabled = !newIndicators[index].enabled
    setIndicators(newIndicators)
    if (stockSymbol) {
      fetchStockData()
    }
  }

  const fetchStockData = async () => {
    if (!stockSymbol) return
    
    setLoading(true)
    try {
      const response = await fetch(api.url(`stock/${stockSymbol}`))
      if (!response.ok) {
        throw new Error('Failed to fetch stock data')
      }
      const data = await response.json()
      setStockData(data)
      
      const enabledIndicators = indicators
        .filter(ind => ind.enabled)
        .map(ind => {
          if (ind.name === 'SMA 20') return 'SMA_20'
          if (ind.name === 'SMA 50') return 'SMA_50'
          if (ind.name === 'EMA 12') return 'EMA_12'
          if (ind.name === 'EMA 26') return 'EMA_26'
          if (ind.name === 'Bollinger Bands') return 'Bollinger'
          if (ind.name === 'Stochastic') return 'Stochastic'
          if (ind.name === 'ADX') return 'ADX'
          return ind.name
        })
        .join(',')
      
      const chartResponse = await fetch(
        api.url(`stock/${stockSymbol}/chart?period=1y&indicators=${enabledIndicators}`)
      )
      if (!chartResponse.ok) {
        throw new Error('Failed to fetch chart data')
      }
      const chartData = await chartResponse.json()
      setChartData(chartData)
    } catch (error) {
      console.error('Error fetching stock data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    fetchStockData()
  }

  const handleStartTrading = () => {
    setShowLanding(false)
  }

  const handleNavigateHome = () => {
    setShowLanding(true)
  }

  if (showLanding) {
    return (
      <div className="min-h-screen">
        <LandingPage 
          onStartTrading={handleStartTrading}
          onShowNewsletter={() => window.location.href = '/newsletter'}
          onNavigateHome={handleNavigateHome}
        />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Toaster position="top-right" />
      
      {/* Header */}
      <header className="bg-slate-900/90 backdrop-blur-xl shadow-2xl sticky top-0 z-50 transition-colors duration-300 border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <Link href="/" className="flex items-center space-x-4 hover:opacity-80 transition-opacity cursor-pointer">
              <div className="flex items-center space-x-4">
                <img src="/assets/logo-color.svg" alt="IterAI Trading" className="h-10 w-10" />
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
                    IterAI Trading
                  </h1>
                  <p className="text-xs text-slate-500 -mt-1">Trading Platform</p>
                </div>
              </div>
            </Link>
            <div className="flex items-center space-x-4">
              <button
                onClick={handleNavigateHome}
                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 transition-all font-medium shadow-lg hover:shadow-xl hover:scale-105 flex items-center gap-2"
              >
                ← <span>Trading Home</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Watchlist Section */}
        <Watchlist 
          key={watchlistState} 
          onSelectStock={(symbol) => {
            setStockSymbol(symbol)
            setTimeout(() => fetchStockData(), 100)
          }}
        />

        {/* Popular Stocks */}
        <PopularStocks onSelectStock={(symbol) => {
          setStockSymbol(symbol)
          setTimeout(() => fetchStockData(), 100)
        }} />

        {/* Search Section */}
        <form onSubmit={handleSearch} className="flex gap-4 mb-6">
          <div className="flex-1">
            <input
              type="text"
              value={stockSymbol}
              onChange={(e) => setStockSymbol(e.target.value.toUpperCase())}
              placeholder="Enter stock symbol (e.g., AAPL, TSLA, MSFT, GOOGL)"
              className="w-full px-6 py-4 rounded-3xl bg-slate-800/60 backdrop-blur-xl text-gray-100 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:bg-slate-800/80 transition-all border border-slate-700/50 text-lg font-medium"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-10 py-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-3xl hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold shadow-lg hover:shadow-xl hover:scale-105 disabled:hover:scale-100"
          >
            {loading ? 'Loading...' : 'Search'}
          </button>
        </form>

        {/* Stock Info Cards */}
        {stockData && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6 animate-slide-up">
              <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg p-5 border border-slate-700/30 hover:border-slate-700/50 transition-all">
                <p className="text-xs text-slate-400 mb-1">Current Price</p>
                <p className="text-2xl font-bold text-white">
                  ${stockData.price.toFixed(2)}
                </p>
              </div>
              <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg p-5 border border-slate-700/30 hover:border-slate-700/50 transition-all">
                <p className="text-xs text-slate-400 mb-1">Change</p>
                <p className={`text-2xl font-bold ${stockData.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {stockData.change >= 0 ? '+' : ''}{stockData.change.toFixed(2)}
                </p>
              </div>
              <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg p-5 border border-slate-700/30 hover:border-slate-700/50 transition-all">
                <p className="text-xs text-slate-400 mb-1">Change %</p>
                <p className={`text-2xl font-bold ${stockData.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {stockData.changePercent >= 0 ? '+' : ''}{stockData.changePercent.toFixed(2)}%
                </p>
              </div>
              <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg p-5 border border-slate-700/30 hover:border-slate-700/50 transition-all">
                <p className="text-xs text-slate-400 mb-1">Volume</p>
                <p className="text-2xl font-bold text-slate-300">
                  {(stockData.volume / 1e6).toFixed(2)}M
                </p>
              </div>
            </div>
            <div className="mb-6 flex justify-end">
              <button
                onClick={() => {
                  const saved = JSON.parse(localStorage.getItem('iterai-watchlist') || '[]')
                  if (!saved.includes(stockData.symbol)) {
                    const updated = [...saved, stockData.symbol]
                    localStorage.setItem('iterai-watchlist', JSON.stringify(updated))
                    setWatchlistState(w => w + 1)
                  }
                }}
                className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl hover:from-purple-600 hover:to-pink-700 transition-all font-medium shadow-lg hover:shadow-xl hover:scale-105 flex items-center gap-2"
              >
                ⭐ Add to Watchlist
              </button>
            </div>
          </>
        )}

        {/* Indicators Toggle */}
        <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg p-6 mb-6 border border-slate-700/50">
          <h2 className="text-lg font-bold mb-4 text-slate-200">Technical Indicators</h2>
          <div className="flex flex-wrap gap-2 mb-4">
            {indicators.map((indicator, index) => (
              <button
                key={indicator.name}
                onClick={() => toggleIndicator(index)}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                  indicator.enabled
                    ? 'bg-blue-600 text-white hover:bg-blue-700'
                    : 'bg-slate-700/40 text-slate-300 hover:bg-slate-700/60 border border-slate-600/30'
                }`}
              >
                {indicator.name}
              </button>
            ))}
          </div>
          
          {/* Indicator Explanations */}
          {indicators.filter(ind => ind.enabled).length > 0 && (
            <div className="bg-slate-900/60 rounded-xl p-4 border border-slate-700/50">
              <p className="text-sm font-semibold text-blue-400 mb-2">Enabled Indicators:</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-slate-300">
                {indicators.filter(ind => ind.enabled).map((indicator, idx) => {
                  const explanations: { [key: string]: string } = {
                    'SMA 20': '20-day Simple Moving Average - Shows short-term price trend',
                    'SMA 50': '50-day Simple Moving Average - Shows medium-term price trend',
                    'EMA 12': '12-day Exponential Moving Average - More responsive to recent price changes',
                    'EMA 26': '26-day Exponential Moving Average - Longer-term EMA for trend analysis',
                    'RSI': 'Relative Strength Index (0-100) - Measures overbought (>70) or oversold (<30) conditions',
                    'MACD': 'Moving Average Convergence Divergence - Shows momentum and trend changes',
                    'Bollinger Bands': 'Price volatility bands - Upper/lower bands show extreme price levels',
                    'Stochastic': 'Stochastic Oscillator - Compares closing price to price range over time',
                    'ADX': 'Average Directional Index - Measures trend strength (not direction)',
                    'Volume': 'Trading volume bars - Shows market activity and conviction'
                  }
                  return (
                    <p key={idx} className="flex items-start gap-2">
                      <span className="text-blue-400 font-medium">{indicator.name}:</span>
                      <span>{explanations[indicator.name] || 'Technical indicator'}</span>
                    </p>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg mb-4 border border-slate-700/50 overflow-hidden">
          <div className="flex flex-wrap">
            <button
              onClick={() => setActiveTab('chart')}
              className={`px-6 py-3 font-semibold transition-all ${
                activeTab === 'chart'
                  ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              Chart & Analysis
            </button>
            <button
              onClick={() => setActiveTab('backtest')}
              className={`px-6 py-3 font-semibold transition-all ${
                activeTab === 'backtest'
                  ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              Backtesting
            </button>
            <button
              onClick={() => setActiveTab('ml')}
              className={`px-6 py-3 font-semibold transition-all ${
                activeTab === 'ml'
                  ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              AI Predictions
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'chart' && chartData && (
          <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg p-5 border border-slate-700/50 overflow-hidden">
            <PlotComponent data={chartData} />
          </div>
        )}

        {activeTab === 'backtest' && stockData && (
          <Backtester symbol={stockData.symbol} />
        )}

        {activeTab === 'ml' && stockData && (
          <MLPredictor symbol={stockData.symbol} />
        )}
      </main>

      {/* Market News Section */}
      {news.length > 0 && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg p-6 border border-slate-700/50">
            <h2 className="text-lg font-bold mb-4 text-slate-200">Market News</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {news.map((item, index) => (
                <a
                  key={index}
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="bg-slate-800/40 backdrop-blur-xl rounded-lg p-4 border border-slate-700/30 hover:bg-slate-800/60 hover:border-slate-700/50 transition-all block group"
                >
                  <div className="flex items-start space-x-2">
                    <div className="flex-shrink-0 w-1.5 h-1.5 bg-blue-500 rounded-full mt-2"></div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-blue-400 mb-1">{item.publisher}</p>
                      <p className="text-slate-200 text-sm font-medium line-clamp-3">{item.title}</p>
                    </div>
                  </div>
                </a>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="bg-slate-900/80 backdrop-blur-xl border-t border-slate-700/50 mt-8 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-slate-400 text-sm">
            © 2024 IterAI. Built for learning and practice.
          </p>
        </div>
      </footer>
    </div>
  )
}

