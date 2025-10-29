import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

// --- ENSURE THIS IS THE VERY FIRST IMPORT ---
import './index.css'
// --- END ENSURE ---

import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)