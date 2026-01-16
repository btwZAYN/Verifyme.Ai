import { useState, useEffect } from 'react';
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react';

const Toast = ({ message, type = 'info', duration = 3000, onClose }) => {
    const [isVisible, setIsVisible] = useState(true);

    useEffect(() => {
        const timer = setTimeout(() => {
            setIsVisible(false);
            setTimeout(onClose, 300);
        }, duration);

        return () => clearTimeout(timer);
    }, [duration, onClose]);

    const icons = {
        success: <CheckCircle size={20} />,
        error: <AlertCircle size={20} />,
        warning: <AlertTriangle size={20} />,
        info: <Info size={20} />
    };

    const colors = {
        success: { bg: 'var(--success-bg)', border: 'var(--success)', text: 'var(--success)' },
        error: { bg: 'var(--error-bg)', border: 'var(--error)', text: 'var(--error)' },
        warning: { bg: 'var(--warning-bg)', border: 'var(--warning)', text: 'var(--warning)' },
        info: { bg: 'var(--info-bg)', border: 'var(--info)', text: 'var(--info)' }
    };

    const style = colors[type] || colors.info;

    return (
        <div
            className={`toast ${isVisible ? 'fade-in' : ''}`}
            style={{
                position: 'fixed',
                top: '2rem',
                right: '2rem',
                zIndex: 9999,
                background: style.bg,
                backdropFilter: 'blur(12px)',
                border: `1px solid ${style.border}`,
                borderRadius: 'var(--radius-md)',
                padding: '1rem 1.5rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                minWidth: '300px',
                maxWidth: '500px',
                boxShadow: 'var(--shadow-lg)',
                opacity: isVisible ? 1 : 0,
                transform: isVisible ? 'translateX(0)' : 'translateX(100%)',
                transition: 'all var(--transition-base)'
            }}
        >
            <div style={{ color: style.text, display: 'flex', alignItems: 'center' }}>
                {icons[type]}
            </div>
            <div style={{ flex: 1, color: 'var(--text-main)', fontSize: '0.95rem' }}>
                {message}
            </div>
            <button
                onClick={() => {
                    setIsVisible(false);
                    setTimeout(onClose, 300);
                }}
                style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                    padding: '0.25rem',
                    display: 'flex',
                    alignItems: 'center',
                    transition: 'color var(--transition-fast)'
                }}
                onMouseEnter={(e) => e.target.style.color = 'var(--text-main)'}
                onMouseLeave={(e) => e.target.style.color = 'var(--text-muted)'}
            >
                <X size={18} />
            </button>
        </div>
    );
};

// Toast Container to manage multiple toasts
export const ToastContainer = ({ toasts, removeToast }) => {
    return (
        <div style={{ position: 'fixed', top: 0, right: 0, zIndex: 9999 }}>
            {toasts.map((toast, index) => (
                <div key={toast.id} style={{ marginTop: index > 0 ? '1rem' : '0' }}>
                    <Toast
                        message={toast.message}
                        type={toast.type}
                        duration={toast.duration}
                        onClose={() => removeToast(toast.id)}
                    />
                </div>
            ))}
        </div>
    );
};

// Hook to use toasts
export const useToast = () => {
    const [toasts, setToasts] = useState([]);

    const addToast = (message, type = 'info', duration = 3000) => {
        const id = Date.now();
        setToasts(prev => [...prev, { id, message, type, duration }]);
    };

    const removeToast = (id) => {
        setToasts(prev => prev.filter(toast => toast.id !== id));
    };

    return { toasts, addToast, removeToast };
};

export default Toast;
