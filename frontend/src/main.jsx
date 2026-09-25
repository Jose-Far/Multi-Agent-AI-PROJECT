import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.jsx';
import AdminApp from './AdminApp.jsx';
import LoginPage from './pages/LoginPage.jsx';
import { AuthProvider, useAuth } from './services/AuthContext.jsx';

function AppGuard() {
  const { user, loading } = useAuth();
  const path = window.location.pathname;

  if (loading) {
    return (
      <div style={{ 
        minHeight: '100vh', 
        backgroundColor: '#05070c', 
        display: 'flex', 
        flexDirection: 'column',
        alignItems: 'center', 
        justifyContent: 'center', 
        color: '#38bdf8', 
        fontFamily: 'monospace',
        gap: '12px'
      }}>
        <div style={{ width: '40px', height: '40px', border: '3px solid rgba(56, 189, 248, 0.2)', borderTopColor: '#38bdf8', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
        <span>INITIALIZING PHISHDEC SOC SESSION...</span>
        <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  // 1. Unauthenticated -> Always Login
  if (!user) {
    return <LoginPage />;
  }

  // 2. Admin & Super Admin Roles Handling
  if (user.role === 'super_admin' || user.role === 'admin') {
    if (path.startsWith('/admin')) {
      return (
        <div className="theme-admin" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
          <AdminApp />
        </div>
      );
    } else {
      return (
        <div className="theme-admin" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
          <App />
        </div>
      );
    }
  }

  // If a normal analyst or viewer tries to access /admin, kick them to root
  if (path.startsWith('/admin')) {
    window.history.replaceState(null, '', '/');
  }

  // 3. Regular Analyst or Viewer

  // 4. Analyst Role -> Tactical Analyst UI
  if (user.role === 'analyst') {
    return (
      <div className="theme-analyst" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <App />
      </div>
    );
  }

  // 4. Viewer Role -> Standard SOC
  return (
    <div className="theme-viewer" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <App />
    </div>
  );
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AuthProvider>
      <AppGuard />
    </AuthProvider>
  </StrictMode>,
);
