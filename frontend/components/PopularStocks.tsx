'use client'

interface PopularStocksProps {
  onSelectStock: (symbol: string) => void
}

const POPULAR_STOCKS = [
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'MSFT', name: 'Microsoft' },
  { symbol: 'GOOGL', name: 'Alphabet' },
  { symbol: 'AMZN', name: 'Amazon' },
  { symbol: 'TSLA', name: 'Tesla' },
  { symbol: 'META', name: 'Meta Platforms' },
  { symbol: 'NVDA', name: 'NVIDIA' },
  { symbol: 'JPM', name: 'JPMorgan Chase' },
  { symbol: 'V', name: 'Visa' },
  { symbol: 'JNJ', name: 'Johnson & Johnson' },
]

export default function PopularStocks({ onSelectStock }: PopularStocksProps) {
  return (
    <div className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-lg p-6 mb-6 border border-slate-700/50">
      <h2 className="text-lg font-bold mb-4 text-slate-200">Popular Stocks</h2>
      <div className="flex flex-wrap gap-2">
        {POPULAR_STOCKS.map((stock) => (
          <button
            key={stock.symbol}
            onClick={() => onSelectStock(stock.symbol)}
            className="px-4 py-2 bg-slate-700/40 text-slate-300 rounded-lg hover:bg-blue-600 hover:text-white transition-all text-sm font-medium border border-slate-600/30"
          >
            {stock.symbol}
          </button>
        ))}
      </div>
    </div>
  )
}

