import { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import { ShieldCheck, Sun, Moon } from 'lucide-react'
import HomePage from './components/HomePage'
import ResultPage from './components/ResultPage'
import './App.css'

function App() {
  const [darkMode, setDarkMode] = useState(false);

  return (
    <div className={`landing-bg${darkMode ? ' dark' : ''}`}>
      <div className="page-content">
        <nav className="navbar">
          <div className="navbar-content">
            <div className="navbar-logo-group">
              <div className="navbar-logo-bg">
                <ShieldCheck size={20} />
              </div>
              <span className="navbar-title">Authentify</span>
            </div>
            <button
              className="theme-toggle nav-toggle"
              onClick={() => setDarkMode(!darkMode)}
              aria-label="Toggle dark mode"
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/result" element={<ResultPage />} />
        </Routes>
      </div>
    </div>
  );
}

export default App
