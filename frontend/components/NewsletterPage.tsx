'use client'

import { useState } from 'react'
import Link from 'next/link'

export default function NewsletterPage({ onBackToPlatform }: { onBackToPlatform: () => void }) {
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="bg-slate-900/80 backdrop-blur-xl shadow-lg sticky top-0 z-50 border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity cursor-pointer">
              <img src="/assets/logo-color.svg" alt="IterAI Newsletter" className="h-8 w-8" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
                IterAI Newsletter
              </h1>
            </Link>
            <button
              onClick={onBackToPlatform}
              className="px-6 py-2 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 transition-all font-semibold shadow-lg hover:shadow-xl hover:scale-105"
            >
              Back to Platform
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-8 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-500 bg-clip-text text-transparent">
            IterAI Newsletter
          </h2>
          <p className="text-xl text-slate-300 mb-6 max-w-3xl mx-auto">
            Discover how AI, Machine Learning, and Cloud technologies are transforming industries
          </p>
          <p className="text-sm text-slate-400 mb-6">
            Get weekly insights on AI tools, ML models, cloud architectures, and real-world use cases
          </p>
        </div>

        {/* Open in New Tab Button */}
        <div className="flex justify-center mb-6">
          <a
            href="https://iterai.beehiiv.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl hover:from-purple-600 hover:to-pink-700 transition-all font-semibold shadow-lg hover:shadow-xl hover:scale-110"
          >
            Open Newsletter in New Tab →
          </a>
        </div>
        
        {/* Newsletter Embed */}
        <div className="bg-white rounded-2xl overflow-hidden shadow-2xl border-4 border-slate-700 animate-slide-up">
          <iframe
            src="https://iterai.beehiiv.com/"
            className="w-full"
            style={{ height: '800px', border: 'none' }}
            title="IterAI Newsletter"
            allow="clipboard-read; clipboard-write"
          />
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8 animate-fade-in-up">
          <div className="bg-gradient-to-br from-blue-900/50 to-indigo-900/50 backdrop-blur-xl rounded-2xl p-6 border border-blue-700/50 hover:scale-105 transition-all">
          <div className="text-4xl mb-3">🤖</div>
          <h3 className="text-xl font-bold text-blue-400 mb-2">AI & ML</h3>
          <p className="text-slate-300 text-sm">
            Learn about AI models, machine learning techniques, and practical applications
          </p>
          </div>
          <div className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 backdrop-blur-xl rounded-2xl p-6 border border-purple-700/50 hover:scale-105 transition-all">
            <div className="text-4xl mb-3">☁️</div>
            <h3 className="text-xl font-bold text-purple-400 mb-2">Cloud Computing</h3>
            <p className="text-slate-300 text-sm">
              Explore cloud platforms, architectures, and deployment strategies
            </p>
          </div>
          <div className="bg-gradient-to-br from-emerald-900/50 to-teal-900/50 backdrop-blur-xl rounded-2xl p-6 border border-emerald-700/50 hover:scale-105 transition-all">
            <div className="text-4xl mb-3">💡</div>
            <h3 className="text-xl font-bold text-emerald-400 mb-2">Use Cases</h3>
            <p className="text-slate-300 text-sm">
              Discover real-world applications of AI, ML, and Cloud technologies
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-slate-900/80 backdrop-blur-xl border-t border-slate-700/50 mt-12 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-slate-400 text-sm">
            © 2024 IterAI. Built for learning and practice.
          </p>
        </div>
      </footer>
    </div>
  )
}

