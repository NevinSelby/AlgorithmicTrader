'use client'

import Link from 'next/link'

export default function CompanyHomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="bg-slate-900/90 backdrop-blur-xl shadow-2xl border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-center space-x-4">
            <img src="/assets/logo-color.svg" alt="IterAI" className="h-16 w-16" />
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-pink-600 bg-clip-text text-transparent">
                IterAI
              </h1>
              <p className="text-sm text-slate-400">Empowering Intelligence</p>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">
            Welcome to <span className="bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">IterAI</span>
          </h2>
          <p className="text-xl text-slate-300 max-w-3xl mx-auto mb-8">
            Your gateway to AI-powered financial intelligence and cutting-edge technology insights
          </p>
        </div>

        {/* Main Cards */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* IterAI Finance Card */}
          <div className="bg-gradient-to-br from-blue-900/20 to-indigo-900/20 backdrop-blur-xl rounded-3xl p-8 border border-blue-700/30 hover:border-blue-500/50 transition-all duration-300 shadow-2xl hover:shadow-blue-900/20 group animate-fade-in-up">
            <div className="flex items-start justify-between mb-6">
              <div className="bg-blue-600 p-4 rounded-2xl group-hover:scale-110 transition-transform">
                <ChartIcon className="h-8 w-8 text-white" />
              </div>
              <span className="px-4 py-1 bg-blue-600/20 text-blue-400 rounded-full text-sm font-semibold">
                Trading Platform
              </span>
            </div>
            <h3 className="text-3xl font-bold text-white mb-4 group-hover:text-blue-400 transition-colors">
              IterAI Trading
            </h3>
            <p className="text-slate-300 mb-6 leading-relaxed">
              Master trading with AI-powered predictions, comprehensive backtesting, and advanced analytics. 
              Explore stocks with 10+ technical indicators, test strategies, and learn algorithmic trading.
            </p>
            
            {/* Features List */}
            <ul className="space-y-3 mb-6">
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>AI-Powered Price Predictions with LSTM models</span>
              </li>
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>10+ Technical Indicators (RSI, MACD, Bollinger Bands, etc.)</span>
              </li>
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>Interactive Charts & Real-Time Market Data</span>
              </li>
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>Strategy Backtesting with Performance Metrics</span>
              </li>
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>Watchlist & Popular Stocks Tracking</span>
              </li>
            </ul>

            <Link
              href="/trading"
              className="w-full py-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 transition-all font-semibold shadow-lg hover:shadow-xl hover:scale-105 flex items-center justify-center gap-2 group"
            >
                <span>Explore IterAI Trading</span>
              <ArrowIcon className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>

          {/* IterAI Newsletter Card */}
          <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 backdrop-blur-xl rounded-3xl p-8 border border-purple-700/30 hover:border-purple-500/50 transition-all duration-300 shadow-2xl hover:shadow-purple-900/20 group animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
            <div className="flex items-start justify-between mb-6">
              <div className="bg-purple-600 p-4 rounded-2xl group-hover:scale-110 transition-transform">
                <NewsletterIcon className="h-8 w-8 text-white" />
              </div>
              <span className="px-4 py-1 bg-purple-600/20 text-purple-400 rounded-full text-sm font-semibold">
                AI & ML Insights
              </span>
            </div>
            <h3 className="text-3xl font-bold text-white mb-4 group-hover:text-purple-400 transition-colors">
              IterAI Newsletter
            </h3>
            <p className="text-slate-300 mb-6 leading-relaxed">
              Stay ahead with the latest insights on Artificial Intelligence, Machine Learning, Cloud Computing, 
              and cutting-edge technologies. Learn how to apply AI solutions in real-world scenarios.
            </p>
            
            {/* Features List */}
            <ul className="space-y-3 mb-6">
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>Weekly AI & Machine Learning Deep Dives</span>
              </li>
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>Cloud Computing Best Practices</span>
              </li>
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>Practical Use Cases & Implementation Guides</span>
              </li>
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>Industry Trends & Technology Updates</span>
              </li>
              <li className="flex items-center text-slate-300">
                <CheckIcon className="h-5 w-5 text-green-400 mr-3 flex-shrink-0" />
                <span>Expert Insights & Community Discussions</span>
              </li>
            </ul>

            <Link
              href="/newsletter"
              className="w-full py-4 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl hover:from-purple-600 hover:to-pink-700 transition-all font-semibold shadow-lg hover:shadow-xl hover:scale-105 flex items-center justify-center gap-2 group"
            >
              <span>Read IterAI Newsletter</span>
              <ArrowIcon className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
        </div>

        {/* Founder Section */}
        <div className="bg-gradient-to-br from-slate-800/60 to-slate-900/60 backdrop-blur-xl rounded-3xl p-8 border border-slate-700/50 animate-fade-in-up mb-12" style={{ animationDelay: '0.2s' }}>
          <div className="flex flex-col md:flex-row items-center md:items-start gap-8">
            <div className="flex-shrink-0">
              <div className="w-32 h-32 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-4xl font-bold text-white shadow-lg">
                NS
              </div>
            </div>
            <div className="flex-1 text-left">
              <h3 className="text-3xl font-bold text-white mb-2">👋 Meet Nevin</h3>
              <p className="text-slate-400 text-lg mb-6">Founder of IterAI | AI Engineer & Data Scientist</p>
              
              <div className="space-y-4 text-slate-300 leading-relaxed">
                <p>
                  I'm Nevin Selby, the founder of IterAI — a company built around one core belief: <span className="text-blue-400 font-semibold">intelligence improves through iteration.</span>
                </p>
                <p>
                  As an AI Engineer and Data Scientist, I specialize in creating scalable, explainable machine learning systems that bridge the gap between research and real-world impact. My work spans NLP, computer vision, and financial AI, from LSTM-based stock prediction models to reinforcement learning trading agents and cloud-native ML pipelines that power real decisions.
                </p>
                <p>
                  At IterAI,                   I'm combining my background in data science, MLOps, and AI engineering with a passion for learning and teaching.
                  Our first projects — IterAI Trading and IterAI Newsletter — reflect that mission: helping people learn, test, and iterate their way to a deeper understanding of AI and the markets.
                </p>
                <p className="text-sm italic text-slate-400">
                  When I'm not experimenting with new models, I share insights and tutorials on machine learning in production, AI for finance, and the human side of automation — always aiming to make technology clearer, faster, and more meaningful.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* About Section */}
        <div className="bg-slate-800/60 backdrop-blur-xl rounded-3xl p-8 border border-slate-700/50 animate-fade-in-up mb-12" style={{ animationDelay: '0.3s' }}>
          <div className="text-center">
            <h3 className="text-2xl font-bold text-white mb-4">About IterAI</h3>
            <p className="text-slate-300 max-w-3xl mx-auto leading-relaxed">
              IterAI is dedicated to democratizing AI and financial intelligence. We provide powerful tools and 
              insights that help individuals and businesses leverage artificial intelligence, machine learning, 
              and algorithmic trading to make informed decisions and stay competitive in today's technology-driven world.
            </p>
          </div>
        </div>

        {/* Mission Statement */}
        <div className="bg-gradient-to-r from-blue-900/20 via-purple-900/20 to-pink-900/20 backdrop-blur-xl rounded-3xl p-8 border border-slate-700/50 animate-fade-in-up mb-12" style={{ animationDelay: '0.4s' }}>
          <div className="text-center">
            <p className="text-2xl text-white mb-4 font-semibold">
              "AI isn't magic — it's a process. One iteration at a time."
            </p>
            <p className="text-slate-400">— Nevin Selby, Founder</p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-slate-400 text-sm">
          <p>© 2024 IterAI. Empowering Intelligence</p>
        </div>
      </div>
    </div>
  )
}

// Icon Components
const ChartIcon = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
)

const NewsletterIcon = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
  </svg>
)

const CheckIcon = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
)

const ArrowIcon = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
  </svg>
)

