'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { api } from '../lib/api'

interface NewsItem {
  title: string
  link: string
  publisher: string
}

interface MarketIndex {
  symbol: string
  name: string
  value: number
  change: number
  changePercent: number
}

interface LandingPageProps {
  onStartTrading: () => void
  onShowNewsletter: () => void
  onNavigateHome?: () => void
}

export default function LandingPage({ onStartTrading, onShowNewsletter, onNavigateHome }: LandingPageProps) {
  const [news, setNews] = useState<NewsItem[]>([])
  const [marketIndices, setMarketIndices] = useState<MarketIndex[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchNews()
    fetchMarketOverview()
  }, [])

  const fetchMarketOverview = async () => {
    try {
      const response = await fetch(api.url('market-overview'))
      if (response.ok) {
        const data = await response.json()
        setMarketIndices(data.overview || [])
      }
    } catch (error) {
      console.error('Error fetching market overview:', error)
    }
  }

  const fetchNews = async () => {
    setLoading(true)
    try {
      const response = await fetch(api.url('news'))
      if (response.ok) {
        const data = await response.json()
        // Extract and format news
        const formattedNews = data.news.slice(0, 6).map((item: any) => ({
          title: item.title || 'Market News',
          link: item.link || '#',
          publisher: item.publisher || 'Financial News'
        }))
        setNews(formattedNews)
      }
    } catch (error) {
      console.error('Error fetching news:', error)
    } finally {
      setLoading(false)
    }
  }

  const ChartIcon = ({ className }: { className?: string }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  )

  const TrendingIcon = ({ className }: { className?: string }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  )

  const CpuIcon = ({ className }: { className?: string }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
    </svg>
  )

  const ArrowPathIcon = ({ className }: { className?: string }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
  )

  const ShieldIcon = ({ className }: { className?: string }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  )

  const CircleStackIcon = ({ className }: { className?: string }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
    </svg>
  )

  const BriefcaseIcon = ({ className }: { className?: string }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
    </svg>
  )

  const AcademicCapIcon = ({ className }: { className?: string }) => (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path d="M12 14l9-5-9-5-9 5 9 5z" />
      <path d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14v9M12 5v9" />
    </svg>
  )

  const features = [
    {
      icon: TrendingIcon,
      title: 'Real-Time Data',
      description: 'Live stock prices, volume, and market data from Yahoo Finance',
      color: 'from-blue-500 to-indigo-600'
    },
    {
      icon: ChartIcon,
      title: 'Technical Indicators',
      description: '10+ indicators including RSI, MACD, Bollinger Bands, and more',
      color: 'from-purple-500 to-pink-600'
    },
    {
      icon: ArrowPathIcon,
      title: 'Strategy Backtesting',
      description: 'Test your trading strategies on historical data with detailed metrics',
      color: 'from-emerald-500 to-teal-600'
    },
    {
      icon: CpuIcon,
      title: 'AI Predictions',
      description: 'Machine learning models predict future price movements',
      color: 'from-violet-500 to-purple-600'
    }
  ]

  const learningTopics = [
    {
      icon: ChartIcon,
      title: 'Technical Analysis',
      description: 'Learn about indicators, charts, and market patterns to make informed trading decisions.',
      topics: ['Moving Averages', 'RSI', 'MACD', 'Candlestick Patterns'],
      color: 'text-blue-400'
    },
    {
      icon: ShieldIcon,
      title: 'Risk Management',
      description: 'Understand position sizing, stop-losses, and portfolio diversification strategies.',
      topics: ['Position Sizing', 'Stop Losses', 'Diversification', 'Risk-Reward Ratio'],
      color: 'text-emerald-400'
    },
    {
      icon: CpuIcon,
      title: 'AI & Machine Learning',
      description: 'Explore how ML models predict stock prices and aid trading decisions.',
      topics: ['LSTM Networks', 'Price Prediction', 'Sentiment Analysis', 'Pattern Recognition'],
      color: 'text-purple-400'
    },
    {
      icon: CircleStackIcon,
      title: 'Market Dynamics',
      description: 'Understand how markets work, news impact, and market cycles.',
      topics: ['Market Cycles', 'News Impact', 'Volatility', 'Trend Analysis'],
      color: 'text-amber-400'
    },
    {
      icon: BriefcaseIcon,
      title: 'Portfolio Management',
      description: 'Learn to build and manage a diversified investment portfolio.',
      topics: ['Asset Allocation', 'Rebalancing', 'Tax Strategies', 'Long-term Investing'],
      color: 'text-cyan-400'
    },
    {
      icon: AcademicCapIcon,
      title: 'Trading Psychology',
      description: 'Master emotional control and develop discipline for successful trading.',
      topics: ['Fear & Greed', 'Discipline', 'Mental Models', 'Execution'],
      color: 'text-rose-400'
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="bg-slate-900/80 backdrop-blur-xl shadow-lg sticky top-0 z-50 border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link 
              href="/"
              className="flex items-center space-x-3 hover:opacity-80 transition-opacity cursor-pointer"
            >
              <img src="/assets/logo-color.svg" alt="IterAI Trading" className="h-8 w-8" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
                IterAI Trading
              </h1>
            </Link>
            <div className="flex gap-3">
              <button
                onClick={onStartTrading}
                className="px-6 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 transition-all font-semibold shadow-lg hover:shadow-xl hover:scale-105 animate-pulse"
              >
                Start Trading
              </button>
              <button
                onClick={onShowNewsletter}
                className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl hover:from-purple-600 hover:to-pink-700 transition-all font-semibold shadow-lg hover:shadow-xl hover:scale-105"
              >
                Newsletter
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-500 bg-clip-text text-transparent animate-fade-in">
              IterAI Finance - Master Trading with AI
            </h1>
            <p className="text-xl text-slate-300 mb-8 max-w-3xl mx-auto animate-fade-in-up">
              Learn, practice, and excel at trading with real-time data, advanced analytics, and AI-powered predictions
            </p>
            <button
              onClick={onStartTrading}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 via-indigo-600 to-purple-600 text-white text-lg rounded-xl hover:from-blue-600 hover:via-indigo-700 hover:to-purple-700 transition-all font-semibold shadow-xl hover:shadow-2xl hover:scale-110 hover:-translate-y-1 animate-bounce-in bg-[length:200%_auto] hover:bg-[length:100%_auto]"
            >
              Start Trading Now →
            </button>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-12">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <div
                  key={index}
                  className="bg-slate-800/60 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 hover:bg-slate-800/80 transition-all hover:scale-105 hover:-translate-y-2 shadow-lg hover:shadow-2xl animate-slide-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="w-12 h-12 mb-4 text-blue-400 transition-transform duration-300 hover:rotate-12 hover:scale-110">
                    <Icon className="w-full h-full" />
                  </div>
                  <h3 className="text-xl font-semibold text-slate-200 mb-2">{feature.title}</h3>
                  <p className="text-slate-400 text-sm">{feature.description}</p>
                </div>
              )
            })}
          </div>
        </div>
      </div>

          {/* Market Overview Section */}
      {marketIndices.length > 0 && (
        <div className="bg-gradient-to-br from-slate-900 to-slate-800 py-8 border-b border-slate-700/50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-2xl font-bold text-center mb-6 text-slate-200">Market Overview</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {marketIndices.map((index, i) => (
                <div key={i} className="bg-slate-800/60 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 hover:bg-slate-800/80 transition-all hover:scale-105 hover:-translate-y-2 shadow-lg hover:shadow-2xl animate-fade-in-up" style={{ animationDelay: `${i * 100}ms` }}>
                  <p className="text-sm text-slate-400 mb-1">{index.name}</p>
                  <p className="text-2xl font-bold text-white">{index.value.toFixed(2)}</p>
                  <p className={`text-sm font-semibold ${index.changePercent >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {index.changePercent >= 0 ? '+' : ''}{index.changePercent.toFixed(2)}%
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Learning Topics Section */}
      <div className="bg-gradient-to-br from-indigo-900 via-purple-900 to-indigo-900 py-12 border-y border-indigo-700/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-8 text-white">Learning Topics</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {learningTopics.map((topic, index) => {
              const Icon = topic.icon
              return (
                <div
                  key={index}
                  className="bg-slate-800/60 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 hover:bg-slate-800/80 transition-all hover:scale-105 hover:-translate-y-2 shadow-lg hover:shadow-2xl animate-slide-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="w-12 h-12 mb-4 transition-transform duration-300 hover:rotate-12 hover:scale-110">
                    <Icon className={`w-full h-full ${topic.color}`} />
                  </div>
                  <h3 className={`text-xl font-semibold mb-2 ${topic.color}`}>{topic.title}</h3>
                  <p className="text-slate-300 mb-4">{topic.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {topic.topics.map((item, i) => (
                      <span
                        key={i}
                        className="px-3 py-1 bg-slate-700/50 rounded-lg text-sm text-slate-300 border border-slate-600/50 hover:bg-slate-600/50 transition-all transform hover:scale-105"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Market News Section */}
      <div className="bg-gradient-to-br from-slate-900 to-slate-800 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-8 text-slate-200">Market News</h2>
          
          {loading ? (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
              <p className="mt-4 text-slate-400">Loading latest news...</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {news.map((item, index) => (
                <a
                  key={index}
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="bg-slate-800/60 backdrop-blur-xl rounded-2xl p-6 border border-slate-700/50 hover:bg-slate-800/80 transition-all hover:scale-105 hover:-translate-y-2 shadow-lg hover:shadow-2xl block animate-fade-in-up"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2 animate-pulse"></div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-blue-400 mb-2">{item.publisher}</p>
                      <p className="text-slate-200 font-medium line-clamp-3">{item.title}</p>
                      <p className="text-xs text-slate-400 mt-3">Read more →</p>
                    </div>
                  </div>
                </a>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-4 text-white">Ready to Start Trading?</h2>
          <p className="text-lg text-slate-300 mb-6 max-w-2xl mx-auto">
            Explore live markets, test strategies, and use AI predictions to make informed decisions
          </p>
          <button
            onClick={onStartTrading}
            className="px-8 py-4 bg-white/10 backdrop-blur-md text-white text-lg rounded-xl hover:bg-white/20 transition-all font-semibold shadow-xl hover:shadow-2xl hover:scale-105 border border-white/20"
          >
            Launch Trading Platform →
          </button>
        </div>
      </div>
    </div>
  )
}

