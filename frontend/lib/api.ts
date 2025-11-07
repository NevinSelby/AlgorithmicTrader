// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Log for debugging (remove in production)
if (typeof window !== 'undefined') {
  console.log('API Base URL:', API_BASE_URL)
}

export const api = {
  baseUrl: API_BASE_URL,
  
  // Helper function to build API URLs
  url: (path: string) => {
    // Remove leading slash if present
    const cleanPath = path.startsWith('/') ? path.slice(1) : path
    const fullUrl = `${API_BASE_URL}/${cleanPath}`
    // Remove any double slashes except after http:// or https://
    const finalUrl = fullUrl.replace(/([^:]\/)\/+/g, '$1')
    return finalUrl
  }
}

