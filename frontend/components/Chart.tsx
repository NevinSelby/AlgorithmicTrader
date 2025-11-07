'use client'

import { useEffect, useRef, useState } from 'react'
import Plotly from 'plotly.js/dist/plotly'

interface ChartProps {
  data: any
}

export default function Chart({ data }: ChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const [isDark, setIsDark] = useState(true)

  useEffect(() => {
    if (data && chartRef.current) {
      const plotData = data.data || []
      const layout = data.layout || {}
      
      // Apply light or dark theme
      if (isDark) {
        // Dark theme
        layout.plot_bgcolor = '#0f172a'
        layout.paper_bgcolor = '#0f172a'
        layout.font = { color: '#ffffff' }
        
        if (layout.xaxis) {
          layout.xaxis.gridcolor = '#1e293b'
          layout.xaxis.title = {...layout.xaxis.title, font: { color: '#ffffff' }}
        }
        if (layout.yaxis) {
          layout.yaxis.gridcolor = '#1e293b'
          layout.yaxis.title = {...layout.yaxis.title, font: { color: '#ffffff' }}
        }
        if (layout.yaxis2) {
          layout.yaxis2.gridcolor = '#1e293b'
        }
        if (layout.yaxis3) {
          layout.yaxis3.gridcolor = '#1e293b'
        }
      } else {
        // Light theme - white background
        layout.plot_bgcolor = '#ffffff'
        layout.paper_bgcolor = '#ffffff'
        layout.font = { color: '#1e293b' }
        
        if (layout.xaxis) {
          layout.xaxis.gridcolor = '#e5e7eb'
          layout.xaxis.title = {...layout.xaxis.title, font: { color: '#1e293b' }}
        }
        if (layout.yaxis) {
          layout.yaxis.gridcolor = '#e5e7eb'
          layout.yaxis.title = {...layout.yaxis.title, font: { color: '#1e293b' }}
        }
        if (layout.yaxis2) {
          layout.yaxis2.gridcolor = '#e5e7eb'
        }
        if (layout.yaxis3) {
          layout.yaxis3.gridcolor = '#e5e7eb'
        }
      }

      Plotly.newPlot(chartRef.current, plotData, layout, {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      })
    }
  }, [data, isDark])

  if (!data) {
    return (
      <div className="w-full h-[500px] flex items-center justify-center bg-slate-800/50 rounded-xl">
        <p className="text-slate-400">No chart data available</p>
      </div>
    )
  }

  return (
    <div className="relative">
      {/* Theme Toggle Button */}
      <div className="absolute top-2 right-2 z-10">
        <button
          onClick={() => setIsDark(!isDark)}
          className="px-3 py-1.5 bg-slate-800/80 backdrop-blur-sm text-white rounded-lg hover:bg-slate-700/80 transition-all text-sm font-medium border border-slate-700/50 shadow-lg"
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark ? '☀️ Light' : '🌙 Dark'}
        </button>
      </div>
      <div ref={chartRef} className="w-full h-[500px]" />
    </div>
  )
}
