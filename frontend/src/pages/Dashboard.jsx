import { useNavigate } from 'react-router-dom';
import { Camera, User, LogOut, Shield, Clock, CheckCircle } from 'lucide-react';
import { useEffect, useState } from 'react';

export default function Dashboard() {
    const navigate = useNavigate();
    const [userName, setUserName] = useState('');

    useEffect(() => {
        const storedName = localStorage.getItem('userName');
        if (!storedName) {
            navigate('/');
        } else {
            setUserName(storedName);
        }
    }, [navigate]);

    const handleLogout = () => {
        localStorage.removeItem('userName');
        navigate('/');
    };

    return (
        <div className="fade-in" style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
            {/* Welcome Header */}
            <div className="glass-card" style={{
                padding: '2.5rem',
                marginBottom: '2rem',
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%)',
                border: '1px solid var(--glass-border)'
            }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                    <div>
                        <h1 style={{ fontSize: '2.5rem', fontWeight: '700', marginBottom: '0.5rem' }}>
                            Welcome back, <span className="text-gradient">{userName}</span>
                        </h1>
                        <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem' }}>
                            Your identity verification dashboard
                        </p>
                    </div>
                    <button
                        className="btn-secondary hover-lift"
                        onClick={handleLogout}
                        style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                    >
                        <LogOut size={18} />
                        Logout
                    </button>
                </div>
            </div>

            {/* Quick Actions */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                gap: '1.5rem',
                marginBottom: '2rem'
            }}>
                <ActionCard
                    icon={<Camera size={32} />}
                    title="Start Verification"
                    description="Begin real-time face verification with your webcam"
                    buttonText="Verify Now"
                    onClick={() => navigate('/verify')}
                    primary
                />
                <ActionCard
                    icon={<User size={32} />}
                    title="Profile"
                    description="View and manage your account information"
                    buttonText="View Profile"
                    onClick={() => { }}
                />
                <ActionCard
                    icon={<Shield size={32} />}
                    title="Security"
                    description="Manage your security settings and preferences"
                    buttonText="Settings"
                    onClick={() => { }}
                />
            </div>

            {/* Stats Overview */}
            <div style={{ marginBottom: '2rem' }}>
                <h2 style={{ fontSize: '1.8rem', fontWeight: '600', marginBottom: '1.5rem' }}>
                    Overview
                </h2>
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                    gap: '1.5rem'
                }}>
                    <StatCard
                        icon={<CheckCircle size={24} />}
                        label="Account Status"
                        value="Active"
                        color="var(--success)"
                    />
                    <StatCard
                        icon={<Clock size={24} />}
                        label="Last Login"
                        value="Just now"
                        color="var(--info)"
                    />
                    <StatCard
                        icon={<Shield size={24} />}
                        label="Security Level"
                        value="High"
                        color="var(--primary)"
                    />
                </div>
            </div>

            {/* Recent Activity */}
            <div className="glass-card" style={{ padding: '2rem' }}>
                <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Clock size={24} />
                    Recent Activity
                </h2>
                <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '3rem 0' }}>
                    <Shield size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
                    <p>No recent verification activity</p>
                    <button
                        className="btn-primary"
                        onClick={() => navigate('/verify')}
                        style={{ marginTop: '1rem' }}
                    >
                        Start Your First Verification
                    </button>
                </div>
            </div>

            {/* Features Info */}
            <div style={{
                marginTop: '2rem',
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                gap: '1.5rem'
            }}>
                <InfoCard
                    title="Real-Time Verification"
                    description="Use your webcam to verify your identity instantly with our advanced facial recognition system."
                />
                <InfoCard
                    title="Secure & Private"
                    description="Your biometric data is encrypted and never shared. We prioritize your privacy and security."
                />
                <InfoCard
                    title="High Accuracy"
                    description="Our AI-powered system provides 99.9% accuracy with lightning-fast processing times."
                />
            </div>
        </div>
    );
}

function ActionCard({ icon, title, description, buttonText, onClick, primary }) {
    return (
        <div className="glass-card hover-lift" style={{ padding: '2rem' }}>
            <div style={{
                display: 'inline-flex',
                padding: '1rem',
                background: primary ? 'var(--primary-light)' : 'var(--glass-bg-light)',
                borderRadius: 'var(--radius-md)',
                marginBottom: '1.5rem',
                color: primary ? 'var(--primary)' : 'var(--text-main)'
            }}>
                {icon}
            </div>
            <h3 style={{ fontSize: '1.3rem', fontWeight: '600', marginBottom: '0.75rem' }}>
                {title}
            </h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', lineHeight: '1.5' }}>
                {description}
            </p>
            <button
                className={primary ? 'btn-primary' : 'btn-secondary'}
                onClick={onClick}
                style={{ width: '100%' }}
            >
                {buttonText}
            </button>
        </div>
    );
}

function StatCard({ icon, label, value, color }) {
    return (
        <div className="glass-card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.75rem' }}>
                <div style={{ color: color }}>
                    {icon}
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                    {label}
                </div>
            </div>
            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: color }}>
                {value}
            </div>
        </div>
    );
}

function InfoCard({ title, description }) {
    return (
        <div className="glass-card" style={{ padding: '1.5rem' }}>
            <h4 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                {title}
            </h4>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.5' }}>
                {description}
            </p>
        </div>
    );
}
