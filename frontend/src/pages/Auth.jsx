import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Camera, Upload, ArrowRight, Loader, CheckCircle, AlertCircle } from 'lucide-react';
import { useToast, ToastContainer } from '../components/Toast';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function Auth({ mode }) {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({ name: '', password: '' });
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const { toasts, addToast, removeToast } = useToast();

    const handleFileChange = (e) => {
        const selected = e.target.files[0];
        if (selected) {
            // Validate file type
            if (!selected.type.startsWith('image/')) {
                addToast('Please select a valid image file', 'error');
                return;
            }
            // Validate file size (max 5MB)
            if (selected.size > 5 * 1024 * 1024) {
                addToast('Image size should be less than 5MB', 'error');
                return;
            }
            setFile(selected);
            setPreview(URL.createObjectURL(selected));
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        // Validation
        if (!formData.name.trim()) {
            addToast('Please enter your name', 'error');
            return;
        }

        if (mode === 'signup' && !file) {
            addToast('Please upload a photo', 'error');
            return;
        }

        setLoading(true);

        try {
            if (mode === 'signup') {
                const data = new FormData();
                data.append('name', formData.name.trim());
                data.append('photo', file);

                const res = await fetch(`${API_URL}/auth/signup`, {
                    method: 'POST',
                    body: data,
                });
                const result = await res.json();

                if (result.success) {
                    addToast('Account created successfully! Please login.', 'success');
                    setTimeout(() => {
                        navigate('/');
                    }, 1500);
                } else {
                    addToast(result.message || 'Signup failed', 'error');
                }
            } else {
                const data = new FormData();
                data.append('name', formData.name.trim());

                const res = await fetch(`${API_URL}/auth/login`, {
                    method: 'POST',
                    body: data
                });
                const result = await res.json();

                if (result.success) {
                    // Store user session
                    localStorage.setItem('userName', formData.name.trim());
                    addToast(`Welcome back, ${formData.name}!`, 'success');
                    setTimeout(() => {
                        navigate('/dashboard');
                    }, 1000);
                } else {
                    addToast(result.message || 'Login failed', 'error');
                }
            }
        } catch (err) {
            console.error('Error:', err);
            addToast('Connection error. Please check if the server is running.', 'error');
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <ToastContainer toasts={toasts} removeToast={removeToast} />
            <div className="fade-in" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
                <div className="glass-card" style={{ width: '100%', maxWidth: '500px', padding: '2.5rem' }}>
                    {/* Header */}
                    <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
                        <div style={{
                            display: 'inline-flex',
                            padding: '1rem',
                            background: 'var(--primary-light)',
                            borderRadius: 'var(--radius-md)',
                            marginBottom: '1rem'
                        }}>
                            {mode === 'signup' ? <Upload size={32} color="var(--primary)" /> : <Camera size={32} color="var(--primary)" />}
                        </div>
                        <h2 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>
                            {mode === 'signup' ? 'Create Account' : 'Welcome Back'}
                        </h2>
                        <p style={{ color: 'var(--text-muted)' }}>
                            {mode === 'signup' ? 'Register your face identity' : 'Login to verify your identity'}
                        </p>
                    </div>

                    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                        {/* Name Input */}
                        <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', fontWeight: '500' }}>
                                Full Name
                            </label>
                            <input
                                type="text"
                                className="input-field"
                                placeholder="e.g. John Doe"
                                value={formData.name}
                                onChange={e => setFormData({ ...formData, name: e.target.value })}
                                required
                                disabled={loading}
                            />
                        </div>

                        {/* Photo Upload for Signup */}
                        {mode === 'signup' && (
                            <div>
                                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.9rem', fontWeight: '500' }}>
                                    Reference Photo
                                </label>
                                <div
                                    onClick={() => !loading && document.getElementById('file-upload').click()}
                                    style={{
                                        border: '2px dashed var(--glass-border)',
                                        borderRadius: 'var(--radius-md)',
                                        padding: '2rem',
                                        textAlign: 'center',
                                        cursor: loading ? 'not-allowed' : 'pointer',
                                        background: 'rgba(15, 23, 42, 0.4)',
                                        transition: 'all var(--transition-base)',
                                        opacity: loading ? 0.6 : 1
                                    }}
                                    onMouseEnter={(e) => !loading && (e.currentTarget.style.borderColor = 'var(--primary)')}
                                    onMouseLeave={(e) => !loading && (e.currentTarget.style.borderColor = 'var(--glass-border)')}
                                >
                                    {preview ? (
                                        <div style={{ position: 'relative' }}>
                                            <img
                                                src={preview}
                                                alt="Preview"
                                                style={{
                                                    width: '100%',
                                                    borderRadius: 'var(--radius-sm)',
                                                    maxHeight: '250px',
                                                    objectFit: 'cover'
                                                }}
                                            />
                                            <div style={{
                                                position: 'absolute',
                                                top: '0.5rem',
                                                right: '0.5rem',
                                                background: 'var(--success)',
                                                borderRadius: 'var(--radius-full)',
                                                padding: '0.5rem',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center'
                                            }}>
                                                <CheckCircle size={20} color="white" />
                                            </div>
                                        </div>
                                    ) : (
                                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', color: 'var(--text-muted)' }}>
                                            <Upload size={40} style={{ marginBottom: '1rem', color: 'var(--primary)' }} />
                                            <span style={{ fontWeight: '500', marginBottom: '0.25rem' }}>Click to upload face photo</span>
                                            <span style={{ fontSize: '0.85rem' }}>PNG, JPG up to 5MB</span>
                                        </div>
                                    )}
                                    <input
                                        id="file-upload"
                                        type="file"
                                        accept="image/*"
                                        onChange={handleFileChange}
                                        style={{ display: 'none' }}
                                        required
                                        disabled={loading}
                                    />
                                </div>
                            </div>
                        )}

                        {/* Submit Button */}
                        <button
                            type="submit"
                            className="btn-primary"
                            disabled={loading}
                            style={{
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'center',
                                gap: '0.5rem',
                                marginTop: '1rem',
                                opacity: loading ? 0.7 : 1,
                                cursor: loading ? 'not-allowed' : 'pointer'
                            }}
                        >
                            {loading ? (
                                <>
                                    <Loader className="spin" size={20} />
                                    {mode === 'signup' ? 'Creating Account...' : 'Logging in...'}
                                </>
                            ) : (
                                <>
                                    {mode === 'signup' ? 'Sign Up' : 'Login'}
                                    <ArrowRight size={20} />
                                </>
                            )}
                        </button>

                        {/* Toggle Link */}
                        <div style={{ textAlign: 'center', marginTop: '0.5rem' }}>
                            <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                                {mode === 'signup' ? 'Already have an account?' : "Don't have an account?"}
                            </span>
                            {' '}
                            <button
                                type="button"
                                onClick={() => navigate(mode === 'signup' ? '/' : '/signup')}
                                disabled={loading}
                                style={{
                                    background: 'transparent',
                                    border: 'none',
                                    color: 'var(--primary)',
                                    cursor: loading ? 'not-allowed' : 'pointer',
                                    fontSize: '0.9rem',
                                    fontWeight: '600',
                                    textDecoration: 'underline',
                                    opacity: loading ? 0.5 : 1
                                }}
                            >
                                {mode === 'signup' ? 'Login' : 'Sign Up'}
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </>
    );
}

