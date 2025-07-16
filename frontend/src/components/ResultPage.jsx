import { useLocation, useNavigate } from 'react-router-dom';
import { AlertCircle, CheckCircle2, XCircle, ArrowLeft } from 'lucide-react';
import './ResultPage.css';

export default function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;
  const error = location.state?.error;

  // Handle error state
  if (error) {
    return (
      <div className="result-container">
        <div className="result-card error-card">
          <div className="result-header error-header">
            <AlertCircle size={28} />
            <h2>Analysis Failed</h2>
          </div>
          <div className="result-content">
            <p className="error-message">
              {error.includes("quota") ? 
                "Service is currently busy. Please try again in a few minutes." :
                "We encountered an issue while analyzing your product. Please try again."}
            </p>
            <button className="action-button" onClick={() => navigate('/')}>
              <ArrowLeft size={18} />
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Handle no result state
  if (!result) {
    return (
      <div className="result-container">
        <div className="result-card">
          <div className="result-header">
            <AlertCircle size={28} />
            <h2>No Results Found</h2>
          </div>
          <div className="result-content">
            <p>No analysis data was found. Please try submitting your product again.</p>
            <button className="action-button" onClick={() => navigate('/')}>
              <ArrowLeft size={18} />
              Go Back
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Handle successful analysis
  const { analysis, recommendations } = result;
  const isAuthentic = analysis.verdict?.toLowerCase() === 'authentic';
  const confidenceScore = typeof analysis.confidence === 'number' ? analysis.confidence : null;

  return (
    <div className="result-container">
      <div className="result-card">
        <div className={`result-header ${isAuthentic ? 'authentic-header' : 'fake-header'}`}>
          {isAuthentic ? <CheckCircle2 size={28} /> : <XCircle size={28} />}
          <h2>Authenticity Analysis</h2>
        </div>
        
        <div className="result-content">
          <div className="verdict-section">
            <span className={`verdict-badge ${isAuthentic ? 'authentic' : 'fake'}`}>
              {analysis.verdict}
            </span>
            {confidenceScore !== null && (
              <span className="confidence-score">
                Confidence: {confidenceScore}%
              </span>
            )}
          </div>

          <div className="analysis-section">
            <h3>Analysis Details</h3>
            <p className="analysis-text">{analysis.full_analysis}</p>
          </div>

          {Array.isArray(recommendations) && recommendations.length > 0 && (
            <div className="recommendations-section">
              <h3>Recommendations</h3>
              <ul className="recommendations-list">
                {recommendations.map((rec, idx) => (
                  <li key={idx} className="recommendation-item">
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button className="action-button" onClick={() => navigate('/')}>
            <ArrowLeft size={18} />
            Verify Another Product
          </button>
        </div>
      </div>
    </div>
  );
} 