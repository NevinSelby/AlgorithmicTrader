'use client'

import { useState } from 'react'

export default function Newsletter() {
  const [height, setHeight] = useState('600px')
  
  return (
    <div className="bg-gradient-to-br from-slate-800 to-slate-900 backdrop-blur-xl rounded-2xl shadow-2xl p-6 animate-slide-up border border-slate-700/50">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2 text-slate-200">IterAI Newsletter</h2>
        <p className="text-sm text-slate-400">
          Stay up-to-date with the latest insights on AI, Data, and Cloud. Subscribe to our newsletter powered by Beehiiv.
        </p>
      </div>
      
      {/* Height Controls */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setHeight('600px')}
          className="px-3 py-1 text-xs bg-slate-700/50 text-slate-300 rounded-lg hover:bg-slate-700 transition-all"
        >
          Standard
        </button>
        <button
          onClick={() => setHeight('800px')}
          className="px-3 py-1 text-xs bg-slate-700/50 text-slate-300 rounded-lg hover:bg-slate-700 transition-all"
        >
          Large
        </button>
        <button
          onClick={() => setHeight('1000px')}
          className="px-3 py-1 text-xs bg-slate-700/50 text-slate-300 rounded-lg hover:bg-slate-700 transition-all"
        >
          Full
        </button>
        <a
          href="https://iterai.beehiiv.com/"
          target="_blank"
          rel="noopener noreferrer"
          className="px-3 py-1 text-xs bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg hover:from-blue-600 hover:to-indigo-700 transition-all ml-auto"
        >
          Open in New Tab →
        </a>
      </div>
      
      {/* Newsletter Embed */}
      <div className="bg-white rounded-xl overflow-hidden shadow-2xl">
        <iframe
          src="https://iterai.beehiiv.com/"
          className="w-full"
          style={{ height }}
          title="IterAI Newsletter"
          allow="clipboard-read; clipboard-write"
        />
      </div>
    </div>
  )
}

