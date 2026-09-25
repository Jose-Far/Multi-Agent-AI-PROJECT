import React, { useState, useEffect } from 'react';
import { getInvestigations, deleteAnalysis, updateInvestigation } from '../services/api';
import { Search, Filter, Shield, AlertTriangle, AlertCircle, CheckCircle, ChevronLeft, ChevronRight, Star, Trash2, ExternalLink } from 'lucide-react';
import InvestigationStats from '../components/InvestigationStats';
import InvestigationFilters from '../components/InvestigationFilters';
import { formatIST } from '../utils/dateUtils';


export default function InvestigationsPage({ onSelectForAnalysis }) {
  const [data, setData] = useState({ analyses: [], stats: {}, total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // State for Day 20 Filtering & Pagination
  const [filters, setFilters] = useState({
    search: '',
    verdict: 'ALL',
    risk_level: 'ALL',
    status: 'ALL',
    date_filter: 'ALL',
    saved_only: false,
    sort: 'newest'
  });
  
  const [page, setPage] = useState(1);
  const LIMIT = 20;

  // Debounce search so we don't spam API
  const [debouncedFilters, setDebouncedFilters] = useState(filters);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedFilters(filters);
      setPage(1); // Reset to page 1 on filter change
    }, 500);
    return () => clearTimeout(timer);
  }, [filters]);

  useEffect(() => {
    loadInvestigations();
  }, [debouncedFilters, page]);

  const loadInvestigations = async () => {
    setLoading(true);
    setError(null);
    try {
      const offset = (page - 1) * LIMIT;
      const res = await getInvestigations({
        ...debouncedFilters,
        limit: LIMIT,
        offset
      });
      setData(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleSave = async (e, id, currentStatus) => {
    e.stopPropagation();
    try {
      await updateInvestigation(id, { is_saved: !currentStatus });
      // Optimistic UI update
      setData(prev => ({
        ...prev,
        analyses: prev.analyses.map(a => a.analysis_id === id ? { ...a, is_saved: !currentStatus } : a)
      }));
    } catch (err) {
      alert("Failed to save investigation");
    }
  };

  const handleDelete = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm(`Delete Investigation ${id}?

This investigation will be permanently removed.`)) return;
    
    try {
      await deleteAnalysis(id);
      loadInvestigations(); // Reload current page
    } catch (err) {
      alert("Failed to delete investigation");
    }
  };

  const getVerdictBadge = (verdict) => {
    const v = verdict ? verdict.toLowerCase() : 'unknown';
    if (v === 'phishing') return <span style={{ color: '#f43f5e', fontWeight: 600 }}>Phishing</span>;
    if (v === 'suspicious') return <span style={{ color: '#fbbf24', fontWeight: 600 }}>Suspicious</span>;
    if (v === 'legitimate') return <span style={{ color: '#10b981', fontWeight: 600 }}>Legitimate</span>;
    return <span style={{ color: 'var(--text-faint)', fontWeight: 600 }}>UNKNOWN</span>;
  };

  const totalPages = Math.ceil((data.total || 0) / LIMIT);

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
      <style>{`
        @media (max-width: 768px) {
          .investigation-table thead {
            display: none;
          }
          .investigation-table, .investigation-table tbody, .investigation-table tr, .investigation-table td {
            display: block;
            width: 100%;
          }
          .investigation-table tr {
            margin-bottom: 16px;
            background-color: rgba(6, 9, 15, 0.4);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 12px;
          }
          .investigation-table td {
            text-align: right;
            padding: 8px 4px !important;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            position: relative;
            padding-left: 50% !important;
          }
          .investigation-table td::before {
            content: attr(data-label);
            position: absolute;
            left: 4px;
            width: 45%;
            text-align: left;
            font-size: 11px;
            color: var(--text-faint);
            text-transform: uppercase;
            font-weight: 600;
          }
          .investigation-table td:last-child {
            border-bottom: 0;
            display: flex;
            justify-content: flex-end;
          }
        }
      `}</style>

      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <h1 style={{ fontSize: '24px', fontWeight: 600, color: '#f8fafc', margin: '0 0 8px 0' }}>
            Investigation History
          </h1>
          <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '14px' }}>
            Manage and review previous security investigations
          </p>
        </div>
      </div>

      <InvestigationStats stats={data.stats} />
      <InvestigationFilters filters={filters} onFilterChange={setFilters} />

      <div style={{
        backgroundColor: 'var(--bg-card)',
        borderRadius: '8px',
        border: '1px solid var(--border-subtle)',
        overflow: 'hidden'
      }}>
        {error ? (
          <div style={{ padding: '40px', textAlign: 'center', color: '#f43f5e' }}>{error}</div>
        ) : loading && data.analyses.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--cyber-blue)' }}>Loading investigations...</div>
        ) : data.analyses.length === 0 ? (
          <div style={{ padding: '60px', textAlign: 'center' }}>
            <Search size={32} color="var(--text-faint)" style={{ marginBottom: '16px' }} />
            <h3 style={{ margin: '0 0 8px 0', color: '#f8fafc', fontWeight: 500 }}>No investigations found</h3>
            <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '14px' }}>Try changing your search or filters.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="investigation-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', backgroundColor: 'rgba(6, 9, 15, 0.4)' }}>
                  <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>*</th>
                  <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Domain</th>
                  <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Verdict</th>
                  <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Risk</th>
                  <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Conf.</th>
                  <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Coverage</th>
                  <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Date</th>
                  <th style={{ padding: '16px', color: 'var(--text-faint)', fontSize: '12px', fontWeight: 600 }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.analyses.map((analysis) => {
                  const isUnknown = analysis.final_verdict === 'unknown' || analysis.risk_score === null;
                  
                  return (
                    <tr 
                      key={analysis.analysis_id}
                      onClick={() => onSelectForAnalysis(analysis.analysis_id)}
                      style={{ 
                        borderBottom: '1px solid var(--border-subtle)',
                        cursor: 'pointer',
                        transition: 'background-color 0.15s ease'
                      }}
                      onMouseOver={e => e.currentTarget.style.backgroundColor = 'rgba(56, 189, 248, 0.05)'}
                      onMouseOut={e => e.currentTarget.style.backgroundColor = 'transparent'}
                    >
                      <td data-label="Save" style={{ padding: '16px' }}>
                        <button 
                          onClick={(e) => handleToggleSave(e, analysis.analysis_id, analysis.is_saved)}
                          style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
                        >
                          <Star size={16} fill={analysis.is_saved ? '#fbbf24' : 'none'} color={analysis.is_saved ? '#fbbf24' : 'var(--text-faint)'} />
                        </button>
                      </td>
                      <td data-label="Domain" style={{ padding: '16px', fontSize: '14px', color: '#f8fafc', fontWeight: 500 }}>
                        {analysis.domain || 'Unknown Domain'}
                        {analysis.status === 'limited_evidence' && (
                          <span style={{ marginLeft: '8px', fontSize: '10px', backgroundColor: 'rgba(251, 191, 36, 0.1)', color: '#fbbf24', padding: '2px 6px', borderRadius: '4px' }}>PARTIAL</span>
                        )}
                        {analysis.tags && (
                          <div style={{ marginTop: '4px', display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                            {(() => {
                              try {
                                const t = JSON.parse(analysis.tags);
                                return t.map(tag => (
                                  <span key={tag} style={{ fontSize: '10px', backgroundColor: 'rgba(56, 189, 248, 0.1)', color: 'var(--cyber-blue)', padding: '2px 6px', borderRadius: '4px' }}>{tag}</span>
                                ));
                              } catch(e) { return null; }
                            })()}
                          </div>
                        )}
                      </td>
                      <td data-label="Verdict" style={{ padding: '16px', fontSize: '13px' }}>
                        {getVerdictBadge(analysis.final_verdict)}
                      </td>
                      <td data-label="Risk" style={{ padding: '16px', fontSize: '13px', color: '#f8fafc', fontFamily: 'monospace' }}>
                        {isUnknown ? '-' : Math.round(analysis.risk_score)}
                      </td>
                      <td data-label="Conf." style={{ padding: '16px', fontSize: '13px', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        {isUnknown ? '0%' : `${Math.round(analysis.confidence * 100)}%`}
                      </td>
                      <td data-label="Coverage" style={{ padding: '16px', fontSize: '13px', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        {analysis.available_agents}/{analysis.total_agents}
                      </td>
                      <td data-label="Date" style={{ padding: '16px', fontSize: '12px', color: 'var(--text-faint)' }}>
                        {formatIST(analysis.created_at).split(' ')[0]}
                      </td>
                      <td data-label="Actions" style={{ padding: '16px' }}>
                        <div style={{ display: 'flex', gap: '12px' }}>
                          <button 
                            onClick={(e) => { e.stopPropagation(); onSelectForAnalysis(analysis.analysis_id); }}
                            style={{ background: 'none', border: 'none', color: 'var(--cyber-blue)', cursor: 'pointer', padding: 0 }}
                            title="Open Investigation"
                          >
                            <ExternalLink size={16} />
                          </button>
                          <button 
                            onClick={(e) => handleDelete(e, analysis.analysis_id)}
                            style={{ background: 'none', border: 'none', color: '#f43f5e', cursor: 'pointer', padding: 0, opacity: 0.7 }}
                            title="Delete Investigation"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        
        {data.total > 0 && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px', borderTop: '1px solid var(--border-subtle)', backgroundColor: 'rgba(6, 9, 15, 0.4)' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              Showing {Math.min((page - 1) * LIMIT + 1, data.total)}-{Math.min(page * LIMIT, data.total)} of {data.total}
            </span>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <button 
                disabled={page === 1}
                onClick={() => setPage(p => p - 1)}
                style={{ background: 'none', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '6px 12px', color: page === 1 ? 'var(--text-faint)' : 'var(--text-muted)', cursor: page === 1 ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <ChevronLeft size={14} /> Prev
              </button>
              
              <span style={{ fontSize: '13px', color: 'var(--text-faint)', margin: '0 8px' }}>
                Page {page} of {totalPages}
              </span>

              <button 
                disabled={page >= totalPages}
                onClick={() => setPage(p => p + 1)}
                style={{ background: 'none', border: '1px solid var(--border-subtle)', borderRadius: '4px', padding: '6px 12px', color: page >= totalPages ? 'var(--text-faint)' : 'var(--text-muted)', cursor: page >= totalPages ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                Next <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}


