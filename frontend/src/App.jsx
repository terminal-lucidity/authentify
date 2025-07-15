import { useState, useRef } from 'react'
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { 
  ShieldCheck, 
  Sun, 
  Moon, 
  UploadCloud, 
  ExternalLink,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Loader2
} from 'lucide-react'
import './App.css'

// Backend API URL - update this based on your environment
const API_URL = 'http://localhost:8000'

function HomePage() {
  const [mode, setMode] = useState('link'); // 'link' or 'image'
  const [productLink, setProductLink] = useState('');
  const [productName, setProductName] = useState('');
  const [productDesc, setProductDesc] = useState('');
  const [productImage, setProductImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef();
  const navigate = useNavigate();

  const handleImageChange = (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      setProductImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setProductImage(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleRemoveImage = () => {
    setProductImage(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (mode === 'link') {
        // Handle link verification
        const response = await fetch(`${API_URL}/scrape_product`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ url: productLink }),
        });

        let data;
        try {
          data = await response.json();
        } catch {
          data = { error: 'Invalid JSON response from server.' };
        }

        if (!response.ok) {
          // Pass error details to result page
          navigate('/result', { state: { error: data?.detail || data?.error || `HTTP error! status: ${response.status}` } });
        } else {
          navigate('/result', { state: { result: data } });
        }
      } else {
        // Handle image verification (future)
        navigate('/result', { state: { result: { analysis: "Image verification coming soon! We're working on it." } } });
      }
    } catch (err) {
      console.error('Error:', err);
      navigate('/result', { state: { error: err.message || 'Failed to verify product. Please try again.' } });
    } finally {
      setLoading(false);
    }
  };

  // Validation for enabling the button
  const canSubmitLink = productLink && !loading;
  // For image form: either (name AND description) OR image is required
  const canSubmitImage = !loading && (
    (productName && productDesc) || // Both text fields filled
    productImage // OR image uploaded
  );

  return (
    <main className="main-section">
      <section className="form-section side-by-side">
        {/* Link Card */}
        <div
          className={`card side-card${mode === 'link' ? ' active' : ' inactive'}`}
          onClick={() => setMode('link')}
          tabIndex={0}
          role="button"
          aria-pressed={mode === 'link'}
        >
          <form
            className="verify-form"
            onSubmit={handleSubmit}
            autoComplete="off"
            style={{ pointerEvents: mode === 'link' ? 'auto' : 'none', opacity: mode === 'link' ? 1 : 0.5 }}
          >
            <label className="input-label">
              Product Link
              <div className="input-with-icon">
                <input
                  type="url"
                  value={productLink}
                  onChange={e => setProductLink(e.target.value)}
                  placeholder="https://example.com/product"
                  required={mode === 'link'}
                  autoFocus={mode === 'link'}
                  disabled={mode !== 'link'}
                />
                <span className="input-icon">
                  <ExternalLink size={18} />
                </span>
              </div>
            </label>
            <button 
              type="submit" 
              className={`verify-btn${loading && mode === 'link' ? ' loading' : ''}`} 
              disabled={!canSubmitLink || mode !== 'link'}
            >
              {loading && mode === 'link' ? (
                <>
                  <Loader2 className="loading-spinner" size={16} />
                  Verifying...
                </>
              ) : 'Verify'}
            </button>
          </form>
          {loading && mode === 'link' && (
            <div className="loading-overlay">
              <div className="loading-text">
                <Loader2 className="loading-spinner" size={18} />
                Analyzing product...
              </div>
            </div>
          )}
        </div>

        {/* Image Card */}
        <div
          className={`card side-card image-card${mode === 'image' ? ' active' : ' inactive'}`}
          onClick={() => setMode('image')}
          tabIndex={0}
          role="button"
          aria-pressed={mode === 'image'}
        >
          <form
            className="verify-form"
            onSubmit={handleSubmit}
            autoComplete="off"
            style={{ pointerEvents: mode === 'image' ? 'auto' : 'none', opacity: mode === 'image' ? 1 : 0.5 }}
          >
            <div className="image-form-layout">
              <div className="image-form-left">
                <input
                  type="text"
                  value={productName}
                  onChange={e => setProductName(e.target.value)}
                  placeholder="Product name"
                  required={mode === 'image' && !productImage}
                  autoFocus={mode === 'image'}
                  disabled={mode !== 'image'}
                  className="form-input"
                />
                <textarea
                  value={productDesc}
                  onChange={e => setProductDesc(e.target.value)}
                  placeholder="Product description"
                  disabled={mode !== 'image'}
                  required={mode === 'image' && !productImage}
                  className="form-input description-input"
                  rows="3"
                />
              </div>
              <div className="image-form-right">
                <div
                  className={`compact-upload${imagePreview ? ' has-image' : ''}`}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  tabIndex={0}
                  aria-label="Product image upload area"
                  onClick={() => mode === 'image' && !imagePreview && fileInputRef.current && fileInputRef.current.click()}
                  role="button"
                  style={{ pointerEvents: mode === 'image' ? 'auto' : 'none' }}
                >
                  {!imagePreview ? (
                    <div className="upload-content">
                      <span className="upload-icon">
                        <UploadCloud size={28} strokeWidth={1.5} />
                      </span>
                      <div className="upload-instructions">
                        <span className="upload-main">Drop image here</span>
                        <span className="upload-or">or <span className="upload-browse">browse files</span></span>
                      </div>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageChange}
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        tabIndex={-1}
                        disabled={mode !== 'image'}
                      />
                    </div>
                  ) : (
                    <div className="image-preview-wrapper">
                      <img src={imagePreview} alt="Preview" className="image-preview compact-preview" />
                      <button 
                        type="button" 
                        className="remove-image-btn compact-remove" 
                        onClick={handleRemoveImage} 
                        aria-label="Remove image"
                        disabled={mode !== 'image'}
                      >
                        ×
                      </button>
                    </div>
                  )}
                </div>
              </div>
    </div>
            <button 
              type="submit" 
              className={`verify-btn${loading && mode === 'image' ? ' loading' : ''}`} 
              disabled={!canSubmitImage || mode !== 'image'}
            >
              {loading && mode === 'image' ? (
                <>
                  <Loader2 className="loading-spinner" size={16} />
                  Verifying...
                </>
              ) : 'Verify'}
    </button>
          </form>
          {loading && mode === 'image' && (
            <div className="loading-overlay">
              <div className="loading-text">
                <Loader2 className="loading-spinner" size={18} />
                Analyzing image...
              </div>
            </div>
          )}
        </div>
      </section>
      {error && (
        <div className="error card">
          <AlertCircle size={20} className="error-icon" />
          {error}
        </div>
      )}
    </main>
  );
}

function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;
  const error = location.state?.error;

  if (error) {
    return (
      <div className="result card">
        <AlertCircle size={20} className="error-icon" />
        <div style={{ color: '#991b1b', marginBottom: '1rem' }}>Error: {typeof error === 'string' ? error : JSON.stringify(error)}</div>
        <button className="return-button" onClick={() => navigate('/')}>Go Back</button>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result card">
        <AlertCircle size={20} className="error-icon" />
        No result data found. <button className="return-button" onClick={() => navigate('/')}>Go Back</button>
      </div>
    );
  }

  // If the backend returned the new structured format
  if (result.verdict || result.full_analysis) {
    return (
      <div className="result card">
        <h2>Authenticity Analysis</h2>
        <div className="verdict-row">
          <span className={`verdict-badge ${result.verdict?.toLowerCase()}`}>{result.verdict}</span>
          {typeof result.confidence === 'number' && (
            <span className="confidence">Confidence: {result.confidence}%</span>
          )}
        </div>
        <pre className="analysis-text">{result.full_analysis}</pre>
        <button className="return-button" onClick={() => navigate('/')}>Verify Another Product</button>
      </div>
    );
  }

  // Fallback for old or error responses
  return (
    <div className="result card">
      <pre>{JSON.stringify(result, null, 2)}</pre>
      <button className="return-button" onClick={() => navigate('/')}>Verify Another Product</button>
    </div>
  );
}

function App() {
  const [darkMode, setDarkMode] = useState(false);
  return (
    <div className={`landing-bg${darkMode ? ' dark' : ''}`}>
      <nav className="navbar">
        <div className="navbar-content">
          <span className="navbar-logo-group">
            <span className="navbar-logo-bg">
              <ShieldCheck size={24} strokeWidth={2.2} />
            </span>
            <span className="navbar-title">Authentify</span>
          </span>
          <button className="theme-toggle nav-toggle" onClick={() => setDarkMode((prev) => !prev)} aria-label="Toggle dark mode">
            {darkMode ? <Moon size={22} /> : <Sun size={22} />}
          </button>
        </div>
      </nav>
      <div className="page-content">
        <header className="hero">
          <h1 className="hero-title">
            Verify Product <span className="highlight">Authenticity</span>
          </h1>
          <p className="hero-subtitle">AI-powered authenticity verification in seconds</p>
          <p className="hero-desc">Upload a product image and paste the link to get results</p>
        </header>
        <div className="hero-divider" />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/result" element={<ResultPage />} />
        </Routes>
      </div>
    </div>
  );
}

export default App
