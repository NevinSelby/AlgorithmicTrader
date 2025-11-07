// API configuration
let API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Ensure the URL is absolute (starts with http:// or https://)
if (API_BASE_URL && !API_BASE_URL.startsWith('http://') && !API_BASE_URL.startsWith('https://')) {
  // If it's a relative path, make it absolute
  if (API_BASE_URL.startsWith('/')) {
    API_BASE_URL = `https://${API_BASE_URL.slice(1)}`
  } else {
    API_BASE_URL = `https://${API_BASE_URL}`
  }
}

// Remove trailing slash
API_BASE_URL = API_BASE_URL.replace(/\/+$/, '')

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
    
    // Log for debugging
    if (typeof window !== 'undefined') {
      console.log('API Call:', finalUrl)
    }
    
    return finalUrl
  }
}

