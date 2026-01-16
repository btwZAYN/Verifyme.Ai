import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { ShieldCheck, User, LogIn, Home as HomeIcon, LayoutDashboard } from 'lucide-react';
import { useState, useEffect } from 'react';
import Auth from './pages/Auth';
import Verification from './pages/Verification';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import ProtectedRoute from './components/ProtectedRoute';
import './index.css';

function Navbar() {
  const location = useLocation();
  const [userName, setUserName] = useState(null);

  useEffect(() => {
    const storedName = localStorage.getItem('userName');
    setUserName(storedName);
  }, [location]);

  const handleLogout = () => {
    localStorage.removeItem('userName');
    setUserName(null);
    window.location.href = '/';
  };

  return (
    <nav className="glass-card fade-in" style={{
      margin: '1rem 2rem',
      padding: '1rem 2rem',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      flexWrap: 'wrap',
      gap: '1rem'
    }}>
      <Link to={userName ? "/dashboard" : "/home"} style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'white' }}>
        <ShieldCheck size={32} color="var(--primary)" />
        <span style={{ fontSize: '1.5rem', fontWeight: '700', letterSpacing: '-0.025em' }}>Verify</span>
      </Link>

      <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
        {userName ? (
          <>
            <NavItem to="/dashboard" current={location.pathname} icon={<LayoutDashboard size={18} />}>Dashboard</NavItem>
            <NavItem to="/verify" current={location.pathname} icon={<ShieldCheck size={18} />}>Verify</NavItem>
            <div style={{
              padding: '0.5rem 1rem',
              background: 'var(--primary-light)',
              borderRadius: 'var(--radius-full)',
              fontSize: '0.9rem',
              fontWeight: '600',
              color: 'var(--primary)'
            }}>
              {userName}
            </div>
            <button
              onClick={handleLogout}
              className="btn-secondary"
              style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }}
            >
              Logout
            </button>
          </>
        ) : (
          <>
            <NavItem to="/home" current={location.pathname} icon={<HomeIcon size={18} />}>Home</NavItem>
            <NavItem to="/" current={location.pathname} icon={<LogIn size={18} />}>Login</NavItem>
            <Link to="/signup" style={{ textDecoration: 'none' }}>
              <button className="btn-primary" style={{ padding: '0.5rem 1.25rem', fontSize: '0.9rem' }}>
                <User size={18} />
                Sign Up
              </button>
            </Link>
          </>
        )}
      </div>
    </nav>
  );
}

function NavItem({ to, current, children, icon }) {
  const isActive = current === to;
  return (
    <Link to={to} style={{
      textDecoration: 'none',
      color: isActive ? 'var(--text-main)' : 'var(--text-muted)',
      fontWeight: isActive ? 600 : 400,
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      transition: 'color var(--transition-base)',
      position: 'relative',
      padding: '0.5rem 0'
    }}>
      {icon}
      {children}
      {isActive && (
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '2px',
          background: 'linear-gradient(90deg, var(--primary) 0%, #a855f7 100%)',
          borderRadius: '2px'
        }} />
      )}
    </Link>
  );
}

function App() {
  return (
    <Router>
      <div className="app-bg">
        <Navbar />
        <div className="page-container">
          <Routes>
            <Route path="/home" element={<Home />} />
            <Route path="/" element={<Auth mode="login" />} />
            <Route path="/signup" element={<Auth mode="signup" />} />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />
            <Route
              path="/verify"
              element={
                <ProtectedRoute>
                  <Verification />
                </ProtectedRoute>
              }
            />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
