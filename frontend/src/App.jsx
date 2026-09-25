import React, { useState, useEffect } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import DashboardPage from './pages/DashboardPage';
import AnalyzePage from './pages/AnalyzePage';
import InvestigationsPage from './pages/InvestigationsPage';
import SettingsPage from './pages/SettingsPage';
import ReportsPage from './pages/ReportsPage';
import ReportPreviewPage from './pages/ReportPreviewPage';
import AdminUsersPage from './pages/admin/AdminUsersPage';
import NotificationHistory from './components/NotificationHistory';
import { checkHealth } from './services/api';

export default function App() {
  const [currentTab, setTab] = useState('dashboard');
  const [healthData, setHealthData] = useState(null);
  const [apiOnline, setApiOnline] = useState(true);
  const [initialUrl, setInitialUrl] = useState('');
  const [targetId, setTargetId] = useState(null);
  const [previewReportId, setPreviewReportId] = useState(null);

  const refreshHealth = async () => {
    try {
      const res = await checkHealth();
      setHealthData(res);
      setApiOnline(res.status === 'healthy');
    } catch {
      setApiOnline(false);
      setHealthData(null);
    }
  };

  useEffect(() => {
    refreshHealth();
    const interval = setInterval(refreshHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleNavigateAnalyze = (url) => {
    setInitialUrl(url || '');
    setTargetId(null);
    setTab('analyze');
  };

  
  const handleViewReport = (id) => {
    setPreviewReportId(id);
  };

  const handleViewInvestigation = (id) => {
    setTargetId(id);
    setTab('analyze');
  };

  return (
    <div className="app-container">
      <Sidebar currentTab={currentTab} setTab={(t) => { setTab(t); setPreviewReportId(null); }} apiOnline={apiOnline} />
      
      <div className="main-content">
        <Header 
          currentTab={currentTab} 
          apiOnline={apiOnline} 
          onRefreshHealth={refreshHealth}
          onNavigate={(tab) => { setTab(tab); setPreviewReportId(null); }}
        />
        
        <main className="content-body">
          {currentTab === 'dashboard' && (
            <DashboardPage onNavigateToAnalyze={handleNavigateAnalyze} onViewInvestigation={handleViewInvestigation} onNavigateToInvestigations={() => setTab('investigations')} />
          )}
          {currentTab === 'analyze' && (
            <AnalyzePage 
              initialUrl={initialUrl} 
              targetId={targetId} 
              onClearTargetId={() => setTargetId(null)}
                onClearInitialUrl={() => setInitialUrl(null)}
              />
          )}
          {currentTab === 'investigations' && (
            <InvestigationsPage 
              onSelectForAnalysis={handleViewInvestigation} 
            />
          )}
          {currentTab === 'reports' && !previewReportId && <ReportsPage onSelectReport={handleViewReport} />}
          {currentTab === 'reports' && previewReportId && <ReportPreviewPage reportId={previewReportId} onBack={() => setPreviewReportId(null)} />}
          {currentTab === 'settings' && (
            <SettingsPage 
              healthData={healthData} 
              apiOnline={apiOnline} 
              onRefresh={refreshHealth} 
            />
          )}
          {currentTab === 'users' && <AdminUsersPage />}
          {currentTab === 'notifications' && (
            <NotificationHistory role="user" />
          )}
        </main>
      </div>
    </div>
  );
}


