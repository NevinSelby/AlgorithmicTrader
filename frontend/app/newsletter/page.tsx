'use client'

import NewsletterPage from '../../components/NewsletterPage'

export default function NewsletterPageRoute() {
  const handleBackToPlatform = () => {
    window.location.href = '/'
  }

  return <NewsletterPage onBackToPlatform={handleBackToPlatform} />
}

