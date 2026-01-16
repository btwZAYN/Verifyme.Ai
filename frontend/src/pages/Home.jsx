import { useNavigate } from 'react-router-dom';
import { ShieldCheck, Camera, Fingerprint, Lock, Zap, Users, ArrowRight, CheckCircle } from 'lucide-react';

export default function Home() {
    const navigate = useNavigate();

    return (
        <div className="fade-in">
            {/* Hero Section */}
            <section style={{
                minHeight: '85vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                textAlign: 'center',
                padding: '2rem',
                position: 'relative',
                overflow: 'hidden'
            }}>
                {/* Animated Background Elements */}
                <div style={{
                    position: 'absolute',
                    top: '10%',
                    left: '10%',
                    width: '300px',
                    height: '300px',
                    background: 'radial-gradient(circle, rgba(99, 102, 241, 0.2) 0%, transparent 70%)',
                    borderRadius: '50%',
                    filter: 'blur(60px)',
                    animation: 'pulse 4s ease-in-out infinite'
                }} />
                <div style={{
                    position: 'absolute',
                    bottom: '10%',
                    right: '10%',
                    width: '400px',
                    height: '400px',
                    background: 'radial-gradient(circle, rgba(168, 85, 247, 0.2) 0%, transparent 70%)',
                    borderRadius: '50%',
                    filter: 'blur(60px)',
                    animation: 'pulse 4s ease-in-out infinite',
                    animationDelay: '2s'
                }} />

                <div style={{ position: 'relative', zIndex: 1, maxWidth: '900px' }}>
                    {/* Logo Icon */}
                    <div style={{
                        display: 'inline-flex',
                        padding: '1.5rem',
                        background: 'var(--primary-light)',
                        borderRadius: 'var(--radius-xl)',
                        marginBottom: '2rem',
                        border: '1px solid var(--glass-border)'
                    }}>
                        <ShieldCheck size={60} color="var(--primary)" />
                    </div>

                    <h1 style={{
                        fontSize: '4rem',
                        fontWeight: '800',
                        marginBottom: '1.5rem',
                        lineHeight: '1.1',
                        letterSpacing: '-0.02em'
                    }}>
                        Secure Identity <br />
                        <span className="text-gradient">Verification System</span>
                    </h1>

                    <p style={{
                        fontSize: '1.25rem',
                        color: 'var(--text-muted)',
                        marginBottom: '3rem',
                        maxWidth: '700px',
                        margin: '0 auto 3rem',
                        lineHeight: '1.6'
                    }}>
                        Advanced facial recognition technology powered by AI. Verify identities in real-time with military-grade security and lightning-fast processing.
                    </p>

                    <div style={{
                        display: 'flex',
                        gap: '1rem',
                        justifyContent: 'center',
                        flexWrap: 'wrap'
                    }}>
                        <button
                            className="btn-primary hover-lift"
                            onClick={() => navigate('/signup')}
                            style={{ fontSize: '1.1rem', padding: '1rem 2rem' }}
                        >
                            Get Started <ArrowRight size={20} />
                        </button>
                        <button
                            className="btn-secondary hover-lift"
                            onClick={() => navigate('/')}
                            style={{ fontSize: '1.1rem', padding: '1rem 2rem' }}
                        >
                            Sign In
                        </button>
                    </div>

                    {/* Stats */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                        gap: '2rem',
                        marginTop: '4rem',
                        maxWidth: '600px',
                        margin: '4rem auto 0'
                    }}>
                        <div>
                            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: 'var(--primary)' }}>99.9%</div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Accuracy</div>
                        </div>
                        <div>
                            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: 'var(--primary)' }}>&lt;1s</div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Processing</div>
                        </div>
                        <div>
                            <div style={{ fontSize: '2.5rem', fontWeight: '700', color: 'var(--primary)' }}>24/7</div>
                            <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Availability</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section style={{ padding: '4rem 2rem', maxWidth: '1200px', margin: '0 auto' }}>
                <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                    <h2 style={{ fontSize: '3rem', fontWeight: '700', marginBottom: '1rem' }}>
                        Why Choose <span className="text-gradient">Verify</span>
                    </h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', maxWidth: '600px', margin: '0 auto' }}>
                        Industry-leading facial recognition technology designed for security and ease of use
                    </p>
                </div>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                    gap: '2rem'
                }}>
                    <FeatureCard
                        icon={<Camera size={32} />}
                        title="Real-Time Verification"
                        description="Instant face recognition with live camera feed. Verify identities in milliseconds with our advanced AI algorithms."
                        delay="0s"
                    />
                    <FeatureCard
                        icon={<Lock size={32} />}
                        title="Military-Grade Security"
                        description="Your biometric data is encrypted and stored securely. We never compromise on privacy and data protection."
                        delay="0.1s"
                    />
                    <FeatureCard
                        icon={<Zap size={32} />}
                        title="Lightning Fast"
                        description="Process thousands of verifications per second. Our optimized system ensures minimal latency and maximum throughput."
                        delay="0.2s"
                    />
                    <FeatureCard
                        icon={<Fingerprint size={32} />}
                        title="High Accuracy"
                        description="99.9% accuracy rate with advanced deep learning models. Continuous improvement through machine learning."
                        delay="0.3s"
                    />
                    <FeatureCard
                        icon={<Users size={32} />}
                        title="Multi-User Support"
                        description="Manage unlimited users with individual profiles. Perfect for organizations of any size."
                        delay="0.4s"
                    />
                    <FeatureCard
                        icon={<ShieldCheck size={32} />}
                        title="Liveness Detection"
                        description="Advanced anti-spoofing technology prevents photo and video attacks. Ensure genuine user presence."
                        delay="0.5s"
                    />
                </div>
            </section>

            {/* How It Works Section */}
            <section style={{ padding: '4rem 2rem', maxWidth: '1200px', margin: '0 auto' }}>
                <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                    <h2 style={{ fontSize: '3rem', fontWeight: '700', marginBottom: '1rem' }}>
                        How It <span className="text-gradient">Works</span>
                    </h2>
                    <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem' }}>
                        Get started in three simple steps
                    </p>
                </div>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                    gap: '3rem'
                }}>
                    <StepCard
                        number="01"
                        title="Sign Up"
                        description="Create your account and upload a reference photo. Our system will securely store your facial biometrics."
                    />
                    <StepCard
                        number="02"
                        title="Login"
                        description="Enter your credentials to access the verification system. Your session is encrypted and secure."
                    />
                    <StepCard
                        number="03"
                        title="Verify"
                        description="Use your webcam for real-time face verification. Get instant results with detailed confidence scores."
                    />
                </div>
            </section>

            {/* CTA Section */}
            <section style={{
                padding: '4rem 2rem',
                textAlign: 'center',
                margin: '4rem auto',
                maxWidth: '800px'
            }}>
                <div className="glass-card" style={{ padding: '3rem', position: 'relative', overflow: 'hidden' }}>
                    <div style={{
                        position: 'absolute',
                        top: '-50%',
                        left: '-50%',
                        width: '200%',
                        height: '200%',
                        background: 'radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%)',
                        animation: 'pulse 3s ease-in-out infinite'
                    }} />

                    <div style={{ position: 'relative', zIndex: 1 }}>
                        <h2 style={{ fontSize: '2.5rem', fontWeight: '700', marginBottom: '1rem' }}>
                            Ready to Get Started?
                        </h2>
                        <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', marginBottom: '2rem' }}>
                            Join thousands of users who trust Verify for their identity verification needs
                        </p>
                        <button
                            className="btn-primary hover-lift"
                            onClick={() => navigate('/signup')}
                            style={{ fontSize: '1.1rem', padding: '1rem 2.5rem' }}
                        >
                            Create Free Account <ArrowRight size={20} />
                        </button>
                    </div>
                </div>
            </section>
        </div>
    );
}

function FeatureCard({ icon, title, description, delay }) {
    return (
        <div
            className="glass-card hover-lift"
            style={{
                padding: '2rem',
                animation: `fadeIn 0.6s ease-out ${delay}`,
                animationFillMode: 'both'
            }}
        >
            <div style={{
                display: 'inline-flex',
                padding: '1rem',
                background: 'var(--primary-light)',
                borderRadius: 'var(--radius-md)',
                marginBottom: '1.5rem',
                color: 'var(--primary)'
            }}>
                {icon}
            </div>
            <h3 style={{ fontSize: '1.4rem', fontWeight: '600', marginBottom: '0.75rem' }}>
                {title}
            </h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: '1.6' }}>
                {description}
            </p>
        </div>
    );
}

function StepCard({ number, title, description }) {
    return (
        <div style={{ textAlign: 'center' }}>
            <div style={{
                fontSize: '4rem',
                fontWeight: '800',
                background: 'linear-gradient(135deg, var(--primary) 0%, #a855f7 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                marginBottom: '1rem',
                opacity: 0.3
            }}>
                {number}
            </div>
            <h3 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '1rem' }}>
                {title}
            </h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: '1.6' }}>
                {description}
            </p>
        </div>
    );
}
