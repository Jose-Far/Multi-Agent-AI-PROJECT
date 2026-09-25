import React, { useState } from 'react';
import { Activity, ShieldAlert, Cpu, X, Database, Zap, Code, Shield } from 'lucide-react';

export default function AdminModelsPage({ healthData }) {
  const agents = healthData?.agents?.list || [];
  const [selectedAgent, setSelectedAgent] = useState(null);

  return (
    <div className="admin-models" style={{ position: 'relative' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '25px' }}>
        <Activity size={28} style={{ color: 'var(--cyber-blue)' }} />
        <h2 style={{ margin: 0, color: 'var(--text-main)' }}>Machine Learning Model Status</h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
        {agents.map((agent, i) => {
           const isFallback = agent.prediction_source.includes('Fallback');
           const isAPI = agent.prediction_source.includes('API');
           
           return (
            <div 
              key={i} 
              onClick={() => setSelectedAgent(agent)}
              style={{ 
                backgroundColor: 'var(--bg-card)', 
                border: `1px solid ${isFallback ? '#fbbf24' : 'var(--border-subtle)'}`, 
                borderRadius: '8px', 
                padding: '20px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: isFallback ? '0 0 10px rgba(251, 191, 36, 0.1)' : 'none'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--cyber-blue)';
                e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = isFallback ? '#fbbf24' : 'var(--border-subtle)';
                e.currentTarget.style.transform = 'none';
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '15px' }}>
                <h3 style={{ margin: 0, fontSize: '16px', color: 'var(--text-main)' }}>{agent.name}</h3>
                {isFallback ? <ShieldAlert size={18} style={{ color: '#fbbf24' }} /> : <Cpu size={18} style={{ color: 'var(--cyber-blue)' }} />}
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Status:</span>
                  <span style={{ color: isFallback ? '#fbbf24' : (isAPI ? '#38bdf8' : '#10b981'), fontWeight: 600, fontSize: '13px' }}>
                    {agent.model}
                  </span>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Prediction Source:</span>
                  <span style={{ color: 'var(--text-main)', fontSize: '13px' }}>{agent.prediction_source}</span>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Feature Dimensions:</span>
                  <span style={{ color: 'var(--text-main)', fontSize: '13px', fontWeight: 'bold' }}>{agent.features}</span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Version:</span>
                  <span style={{ color: 'var(--text-faint)', fontSize: '13px' }}>{agent.version}</span>
                </div>
              </div>
            </div>
          )
        })}
        {agents.length === 0 && (
          <div style={{ color: 'var(--text-faint)', padding: '20px' }}>API is offline, cannot load models.</div>
        )}
      </div>

      {/* DETAILED POPUP MODAL */}
      {selectedAgent && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(9, 9, 14, 0.85)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 9999
        }}>
          <div style={{
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--cyber-blue)',
            borderRadius: '12px',
            width: '100%',
            maxWidth: '600px',
            padding: '30px',
            position: 'relative',
            boxShadow: '0 0 30px rgba(0, 240, 255, 0.15)'
          }}>
            <button 
              onClick={() => setSelectedAgent(null)}
              style={{
                position: 'absolute', top: '20px', right: '20px',
                background: 'transparent', border: 'none',
                color: 'var(--text-muted)', cursor: 'pointer'
              }}
            >
              <X size={24} />
            </button>

            <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '25px' }}>
              <Cpu size={32} style={{ color: 'var(--cyber-blue)' }} />
              <div>
                <h2 style={{ margin: 0, fontSize: '24px', color: '#fff' }}>{selectedAgent.name} - Detailed Profile</h2>
                <span style={{ color: 'var(--cyber-blue)', fontSize: '14px', fontWeight: 600 }}>v{selectedAgent.version} | {selectedAgent.prediction_source}</span>
              </div>
            </div>

            {selectedAgent.details ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                
                <div style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)', padding: '15px', borderRadius: '8px' }}>
                  <h4 style={{ margin: '0 0 8px 0', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Shield size={16} style={{ color: '#a78bfa' }} /> Purpose & Objective
                  </h4>
                  <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '14px', lineHeight: '1.5' }}>
                    {selectedAgent.details.purpose}
                  </p>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                  <div style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)', padding: '15px', borderRadius: '8px' }}>
                    <h4 style={{ margin: '0 0 8px 0', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Code size={16} style={{ color: '#34d399' }} /> Core Algorithm
                    </h4>
                    <span style={{ color: 'var(--text-muted)', fontSize: '14px' }}>{selectedAgent.details.algorithm}</span>
                  </div>

                  <div style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)', padding: '15px', borderRadius: '8px' }}>
                    <h4 style={{ margin: '0 0 8px 0', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Zap size={16} style={{ color: '#fbbf24' }} /> Latency Profile
                    </h4>
                    <span style={{ color: 'var(--text-muted)', fontSize: '14px' }}>{selectedAgent.details.latency_profile}</span>
                  </div>
                </div>

                <div style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)', padding: '15px', borderRadius: '8px' }}>
                  <h4 style={{ margin: '0 0 8px 0', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Database size={16} style={{ color: '#38bdf8' }} /> Data Vectors & Dependencies
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Dimensions:</span>
                      <span style={{ color: 'var(--text-main)', fontSize: '13px', fontWeight: 'bold' }}>{selectedAgent.features} Features</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Data Source:</span>
                      <span style={{ color: 'var(--text-main)', fontSize: '13px' }}>{selectedAgent.details.data_sources}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Dependencies:</span>
                      <span style={{ color: 'var(--text-main)', fontSize: '13px' }}>{selectedAgent.details.dependencies}</span>
                    </div>
                  </div>
                </div>

              </div>
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-faint)' }}>
                No detailed profile information available for this agent.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
