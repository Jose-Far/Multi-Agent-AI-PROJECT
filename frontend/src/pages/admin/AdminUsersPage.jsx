import React, { useState, useEffect } from 'react';
import { Users, UserPlus, Shield, CheckCircle, Ban, Key, RefreshCw, X, AlertCircle, Trash2 } from 'lucide-react';
import { getAdminUsers, createAdminUser, updateAdminUser, resetUserPassword, deleteAdminUser } from '../../services/api';
import { useAuth } from '../../services/AuthContext';
import { formatIST } from '../../utils/dateUtils';

const ROLE_COLORS = {
  super_admin: { bg: 'rgba(217, 70, 239, 0.15)', text: '#e879f9', border: '#d946ef' },
  admin: { bg: 'rgba(244, 63, 94, 0.15)', text: '#fb7185', border: '#f43f5e' },
  analyst: { bg: 'rgba(56, 189, 248, 0.15)', text: '#38bdf8', border: '#38bdf8' },
  viewer: { bg: 'rgba(251, 191, 36, 0.15)', text: '#fbbf24', border: '#fbbf24' }
};

export default function AdminUsersPage() {
  const { user: currentUser } = useAuth();
  const isSuperAdmin = currentUser?.role === 'super_admin';

  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Add User Modal State
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newEmail, setNewEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newRole, setNewRole] = useState('analyst');
  const [submitting, setSubmitting] = useState(false);

  // Reset Password Modal State
  const [resetUserId, setResetUserId] = useState(null);
  const [resetPasswordVal, setResetPasswordVal] = useState('');

  const loadUsers = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getAdminUsers();
      setUsers(data.users || []); console.log('Fetched users:', data.users);
    } catch (err) {
      setError(err.message || 'Failed to load users');
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    loadUsers();
    // Auto-refresh users every 5 seconds to show real-time online status
    const interval = setInterval(() => {
      getAdminUsers().then(data => {
        setUsers(data.users || []);
      }).catch(err => console.error("Polling error:", err));
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleCreateUser = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await createAdminUser({
        username: newUsername,
        email: newEmail,
        password: newPassword,
        role: newRole
      });
      setSuccessMsg(`User ${newUsername} created successfully.`);
      setIsAddOpen(false);
      setNewUsername('');
      setNewEmail('');
      setNewPassword('');
      loadUsers();
    } catch (err) {
      setError(err.message || 'Failed to create user');
    } finally {
      setSubmitting(false);
    }
  };

  const handleRoleChange = async (userId, role) => {
    try {
      await updateAdminUser(userId, { role });
      setSuccessMsg('User role updated successfully.');
      loadUsers();
    } catch (err) {
      setError(err.message || 'Failed to update role');
    }
  };

  const handleDeleteUser = async (userId, username) => {
    if (!window.confirm(`Are you sure you want to permanently delete user '${username}'?\nThis action cannot be undone.`)) return;
    try {
      await deleteAdminUser(userId);
      setSuccessMsg(`User ${username} was permanently deleted.`);
      loadUsers();
    } catch (err) {
      setError(err.message || 'Failed to delete user');
    }
  };

  const handleToggleStatus = async (userId, currentStatus) => {
    const nextStatus = currentStatus === 'active' ? 'disabled' : 'active';
    try {
      await updateAdminUser(userId, { status: nextStatus });
      setSuccessMsg(`User account ${nextStatus} successfully.`);
      loadUsers();
    } catch (err) {
      setError(err.message || 'Failed to toggle status');
    }
  };

  const handleResetPasswordSubmit = async (e) => {
    e.preventDefault();
    if (!resetPasswordVal || resetPasswordVal.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    try {
      await resetUserPassword(resetUserId, resetPasswordVal);
      setSuccessMsg('Password has been reset successfully.');
      setResetUserId(null);
      setResetPasswordVal('');
    } catch (err) {
      setError(err.message || 'Failed to reset password');
    }
  };

  return (
    <div className="admin-users" style={{ maxWidth: '1200px', margin: '0 auto', paddingTop: '10px' }}>
      {/* Top Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Users size={26} style={{ color: 'var(--cyber-blue)' }} />
            <h2 style={{ margin: 0, color: 'var(--text-main)', fontSize: '20px', fontWeight: 700 }}>
              User Management & Access Control
            </h2>
          </div>
          <p style={{ margin: '4px 0 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            Manage identity, role assignments (RBAC), and account lifecycle across the platform.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button 
            onClick={loadUsers} 
            style={{
              background: 'var(--bg-card)', 
              border: '1px solid var(--border-subtle)', 
              color: 'var(--text-main)', 
              borderRadius: '6px', 
              padding: '8px 14px', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '6px', 
              cursor: 'pointer',
              fontSize: '13px'
            }}
          >
            <RefreshCw size={14} /> Refresh
          </button>

          <button 
            onClick={() => setIsAddOpen(true)}
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px', 
              backgroundColor: 'var(--cyber-blue)', 
              color: '#fff', 
              border: 'none', 
              padding: '8px 18px', 
              borderRadius: '6px', 
              fontWeight: 600, 
              fontSize: '13px',
              cursor: 'pointer',
              boxShadow: '0 0 15px rgba(56, 189, 248, 0.3)'
            }}
          >
            <UserPlus size={16} /> Add User
          </button>
        </div>
      </div>

      {/* Notifications */}
      {error && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 16px', backgroundColor: 'rgba(244, 63, 94, 0.12)', border: '1px solid rgba(244, 63, 94, 0.3)', borderRadius: '8px', color: '#fb7185', marginBottom: '20px', fontSize: '13px' }}>
          <AlertCircle size={18} />
          <span style={{ flex: 1 }}>{error}</span>
          <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: '#fb7185', cursor: 'pointer' }}><X size={16} /></button>
        </div>
      )}
      {successMsg && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 16px', backgroundColor: 'rgba(16, 185, 129, 0.12)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '8px', color: '#34d399', marginBottom: '20px', fontSize: '13px' }}>
          <CheckCircle size={18} />
          <span style={{ flex: 1 }}>{successMsg}</span>
          <button onClick={() => setSuccessMsg(null)} style={{ background: 'none', border: 'none', color: '#34d399', cursor: 'pointer' }}><X size={16} /></button>
        </div>
      )}

      {/* Users Table */}
      <div style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: '8px', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ backgroundColor: 'var(--bg-card-elevated)', borderBottom: '1px solid var(--border-subtle)' }}>
              <th style={{ padding: '14px 20px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '12px', textTransform: 'uppercase' }}>User</th>
              <th style={{ padding: '14px 20px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '12px', textTransform: 'uppercase' }}>Email</th>
              <th style={{ padding: '14px 20px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '12px', textTransform: 'uppercase' }}>Role (RBAC)</th>
              <th style={{ padding: '14px 20px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '12px', textTransform: 'uppercase' }}>Status</th>
              <th style={{ padding: '14px 20px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '12px', textTransform: 'uppercase' }}>Last Login</th>
              <th style={{ padding: '14px 20px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '12px', textTransform: 'uppercase', textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan="6" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-faint)' }}>Loading users from database...</td>
              </tr>
            ) : users.length === 0 ? (
              <tr>
                <td colSpan="6" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-faint)' }}>No users found.</td>
              </tr>
            ) : (
              users.map(u => {
                const roleStyle = ROLE_COLORS[u.role] || ROLE_COLORS.analyst;
                const isActive = u.status === 'active';

                return (
                  <tr key={u.user_id} className="user-row" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                    <td style={{ padding: '14px 20px', color: 'var(--text-main)', fontWeight: 600 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{ 
                          width: '32px', height: '32px', borderRadius: '50%', 
                          backgroundColor: roleStyle.bg, border: `1px solid ${roleStyle.border}`,
                          display: 'flex', alignItems: 'center', justifyContent: 'center' 
                        }}>
                          {(u.role === 'admin' || u.role === 'super_admin') ? <Shield size={16} color={roleStyle.text} /> : <Users size={16} color={roleStyle.text} />}
                        </div>
                        <div>
                          <div>{u.username}</div>
                          <div style={{ fontSize: '11px', color: 'var(--text-faint)' }}>{u.user_id}</div>
                        </div>
                      </div>
                    </td>

                    <td style={{ padding: '14px 20px', color: 'var(--text-muted)', fontSize: '13px' }}>
                      {u.email}
                    </td>

                    <td style={{ padding: '14px 20px' }}>
                      <select
                        value={u.role}
                        onChange={(e) => handleRoleChange(u.user_id, e.target.value)}
                        disabled={!isSuperAdmin && (u.role === 'super_admin' || u.role === 'admin')}
                        style={{
                          backgroundColor: 'var(--bg-dark)',
                          border: `1px solid ${roleStyle.border}`,
                          color: roleStyle.text,
                          padding: '4px 8px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          fontWeight: 600,
                          cursor: (!isSuperAdmin && (u.role === 'super_admin' || u.role === 'admin')) ? 'not-allowed' : 'pointer',
                          opacity: (!isSuperAdmin && (u.role === 'super_admin' || u.role === 'admin')) ? 0.6 : 1
                        }}
                      >
                        {isSuperAdmin && <option value="super_admin">Super Admin</option>}
                        <option value="admin">Administrator</option>
                        <option value="analyst">Analyst</option>
                        <option value="viewer">Viewer</option>
                      </select>
                    </td>

                    <td style={{ padding: '14px 20px' }}>
                      <span style={{ 
                        padding: '4px 10px', borderRadius: '12px', fontSize: '11px', fontWeight: 700,
                        textTransform: 'uppercase',
                        backgroundColor: isActive ? 'rgba(16, 185, 129, 0.12)' : 'rgba(244, 63, 94, 0.12)',
                        color: isActive ? '#34d399' : '#fb7185',
                        border: `1px solid ${isActive ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`
                      }}>
                        {u.status}
                      </span>
                    </td>

                    <td style={{ padding: '14px 20px', color: 'var(--text-faint)', fontSize: '12px' }}>
                      {u.is_online ? (
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                          <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#34d399', boxShadow: '0 0 8px rgba(52, 211, 153, 0.6)' }}></span>
                          <span style={{ color: '#34d399', fontWeight: 600 }}>Online</span>
                        </div>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Offline</span>
                          <span style={{ fontSize: '11px', color: 'var(--text-faint)' }}>{u.last_login ? formatIST(u.last_login) : 'Never logged in'}</span>
                        </div>
                      )}
                    </td>

                    <td style={{ padding: '14px 20px', textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '8px' }}>
                        <button onClick={() => setResetUserId(u.user_id)} title="Reset User Password" className="user-action-btn btn-reset">
                            <Key size={12} />
                            <span>Reset</span>
                          </button>

                        <button onClick={() => handleToggleStatus(u.user_id, u.status)} title={isActive ? "Disable User Account" : "Enable User Account"} className={`user-action-btn ${isActive ? 'btn-disable' : 'btn-enable'}`}>
                            {isActive ? <Ban size={12} /> : <CheckCircle size={12} />}
                            <span>{isActive ? 'Disable' : 'Enable'}</span>
                          </button>
                      
                          {isSuperAdmin && (
                            <button onClick={() => handleDeleteUser(u.user_id, u.username)} title="Delete User Account" className="user-action-btn btn-delete">
                                <Trash2 size={12} />
                                <span>Delete</span>
                              </button>
                          )}
                        </div>
                      </td>
                    </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Add User Modal */}
      {isAddOpen && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.75)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
        }}>
          
            <div style={{
              width: '100%', maxWidth: '420px', backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)', borderRadius: '12px',
              padding: '24px', boxShadow: '0 20px 40px rgba(0,0,0,0.6)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
                <h3 style={{ margin: 0, fontSize: '16px', color: 'var(--text-main)' }}>Create Platform User</h3>
                <button onClick={() => setIsAddOpen(false)} style={{ background: 'none', border: 'none', color: 'var(--text-faint)', cursor: 'pointer' }}><X size={18} /></button>
              </div>

              {error && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px 14px', backgroundColor: 'rgba(244, 63, 94, 0.12)', border: '1px solid rgba(244, 63, 94, 0.3)', borderRadius: '6px', color: '#fb7185', marginBottom: '16px', fontSize: '12px' }}>
                  <AlertCircle size={16} />
                  <span style={{ flex: 1 }}>{error}</span>
                </div>
              )}

              <form onSubmit={handleCreateUser}>
              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>Username</label>
                <input 
                  type="text" required value={newUsername} onChange={e => setNewUsername(e.target.value)}
                  placeholder="e.g. analyst03"
                  style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-subtle)', borderRadius: '6px', color: '#fff' }}
                />
              </div>

              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>Email</label>
                <input 
                  type="email" required value={newEmail} onChange={e => setNewEmail(e.target.value)}
                  placeholder="analyst03@phishdec.local"
                  style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-subtle)', borderRadius: '6px', color: '#fff' }}
                />
              </div>

              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>Initial Password</label>
                <input 
                  type="password" required minLength="6" value={newPassword} onChange={e => setNewPassword(e.target.value)}
                  placeholder="Minimum 6 characters"
                  style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-subtle)', borderRadius: '6px', color: '#fff' }}
                />
              </div>

              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>Role</label>
                <select 
                  value={newRole} onChange={e => setNewRole(e.target.value)}
                  style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-subtle)', borderRadius: '6px', color: '#fff' }}
                >
                  {isSuperAdmin && <option value="super_admin">Super Admin (Platform Owner)</option>}
                  {isSuperAdmin && <option value="admin">Administrator (SOC Admin)</option>}
                  <option value="analyst">Analyst (SOC Operator)</option>
                  <option value="viewer">Viewer (Read-Only)</option>
                </select>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button type="button" onClick={() => setIsAddOpen(false)} style={{ padding: '8px 14px', background: 'transparent', border: '1px solid var(--border-subtle)', color: 'var(--text-muted)', borderRadius: '6px', cursor: 'pointer' }}>Cancel</button>
                <button type="submit" disabled={submitting} style={{ padding: '8px 16px', backgroundColor: 'var(--cyber-blue)', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: 600, cursor: 'pointer' }}>
                  {submitting ? 'Creating...' : 'Create Account'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Reset Password Modal */}
      {resetUserId && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.75)', backdropFilter: 'blur(4px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
        }}>
          
            <div style={{
              width: '100%', maxWidth: '380px', backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-subtle)', borderRadius: '12px',
              padding: '24px', boxShadow: '0 20px 40px rgba(0,0,0,0.6)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h3 style={{ margin: 0, fontSize: '15px', color: 'var(--text-main)' }}>Reset Password</h3>
                <button onClick={() => setResetUserId(null)} style={{ background: 'none', border: 'none', color: 'var(--text-faint)', cursor: 'pointer' }}><X size={16} /></button>
              </div>

              {error && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '10px 14px', backgroundColor: 'rgba(244, 63, 94, 0.12)', border: '1px solid rgba(244, 63, 94, 0.3)', borderRadius: '6px', color: '#fb7185', marginBottom: '16px', fontSize: '12px' }}>
                  <AlertCircle size={16} />
                  <span style={{ flex: 1 }}>{error}</span>
                </div>
              )}

              <form onSubmit={handleResetPasswordSubmit}>
              <div style={{ marginBottom: '18px' }}>
                <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '6px' }}>New Password</label>
                <input 
                  type="password" required minLength="6" value={resetPasswordVal} onChange={e => setResetPasswordVal(e.target.value)}
                  placeholder="Enter new password"
                  style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', backgroundColor: 'var(--bg-dark)', border: '1px solid var(--border-subtle)', borderRadius: '6px', color: '#fff' }}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button type="button" onClick={() => setResetUserId(null)} style={{ padding: '8px 14px', background: 'transparent', border: '1px solid var(--border-subtle)', color: 'var(--text-muted)', borderRadius: '6px', cursor: 'pointer' }}>Cancel</button>
                <button type="submit" style={{ padding: '8px 16px', backgroundColor: 'var(--cyber-blue)', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: 600, cursor: 'pointer' }}>Reset Password</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
