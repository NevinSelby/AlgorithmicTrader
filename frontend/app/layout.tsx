import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'IterAI - AI-Powered Trading Platform',
  description: 'IterAI Trading: Master trading with AI-powered predictions, backtesting, and advanced analytics',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="dark bg-slate-950">
        {children}
      </body>
    </html>
  )
}

