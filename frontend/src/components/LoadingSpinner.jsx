import { Loader } from 'lucide-react';

export default function LoadingSpinner({ size = 40, color = 'var(--primary)' }) {
    return (
        <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '2rem'
        }}>
            <Loader
                size={size}
                color={color}
                className="spin"
            />
        </div>
    );
}

export function FullPageLoader() {
    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            background: 'rgba(15, 23, 42, 0.9)',
            backdropFilter: 'blur(8px)',
            zIndex: 9999
        }}>
            <div style={{ textAlign: 'center' }}>
                <Loader size={60} color="var(--primary)" className="spin" />
                <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Loading...</p>
            </div>
        </div>
    );
}
