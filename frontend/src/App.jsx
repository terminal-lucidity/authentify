import { useState, useRef } from 'react'
import { ShieldCheck, Sun, Moon, UploadCloud, ExternalLink } from 'lucide-react'
import './App.css'

function App() {
  const [productLink, setProductLink] = useState('');
  const [productImage, setProductImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [darkMode, setDarkMode] = useState(false);
  const fileInputRef = useRef();

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
    setResult(null);
    // TODO: Replace with actual backend call
    setTimeout(() => {
      setResult('This is a placeholder result.');
      setLoading(false);
    }, 1500);
  };

  const handleThemeToggle = () => {
    setDarkMode((prev) => !prev);
  };

  const canSubmit = productLink && productImage && !loading;

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
          <button className="theme-toggle nav-toggle" onClick={handleThemeToggle} aria-label="Toggle dark mode">
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
          <p className="hero-desc">Upload a product image and paste the link to get instant results</p>
        </header>
        <main className="main-section">
          <section className="form-section">
            <form className="verify-form card" onSubmit={handleSubmit} autoComplete="off">
              <label className="input-label">
                Product Link
                <div className="input-with-icon">
                  <input
                    type="url"
                    value={productLink}
                    onChange={e => setProductLink(e.target.value)}
                    placeholder="https://example.com/product"
                    required
                    autoFocus
                  />
                  <span className="input-icon">
                    <ExternalLink size={18} />
                  </span>
                </div>
              </label>
              <div
                className={`upload-zone${imagePreview ? ' has-image' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                tabIndex={0}
                aria-label="Product image upload area"
                onClick={() => !imagePreview && fileInputRef.current && fileInputRef.current.click()}
                role="button"
              >
                {!imagePreview ? (
                  <div className="upload-content">
                    <UploadCloud size={40} strokeWidth={2.2} />
                    <div className="upload-instructions">
                      <span className="upload-main">Drop your image here</span>
                      <span className="upload-or"> or <span className="upload-browse">browse files</span></span>
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageChange}
                      ref={fileInputRef}
                      style={{ display: 'none' }}
                      tabIndex={-1}
                    />
                  </div>
                ) : (
                  <div className="image-preview-wrapper">
                    <img src={imagePreview} alt="Preview" className="image-preview" />
                    <button type="button" className="remove-image-btn" onClick={handleRemoveImage} aria-label="Remove image">
                      Remove
                    </button>
                  </div>
                )}
              </div>
              <button type="submit" className="verify-btn" disabled={!canSubmit}>
                {loading ? 'Verifying...' : 'Verify Authenticity'}
              </button>
            </form>
            {error && <div className="error card">{error}</div>}
            {result && <div className="result card">{result}</div>}
          </section>
        </main>
      </div>
    </div>
  );
}

export default App
