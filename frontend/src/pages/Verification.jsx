import { useRef, useEffect, useState } from 'react';
import { Camera, CheckCircle, XCircle, AlertTriangle, Wifi, WifiOff, Play, Pause } from 'lucide-react';
import { useToast, ToastContainer } from '../components/Toast';

// WebSocket URL - uses environment variable in production
const WS_BASE = import.meta.env.VITE_WS_URL || `ws://${window.location.hostname}:8000`;
const SOCKET_URL = `${WS_BASE}/ws/verify`;

export default function Verification() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const ws = useRef(null);
    const reconnectTimer = useRef(null);

    const [status, setStatus] = useState("Initializing...");
    const [results, setResults] = useState([]);
    const [systemActive, setSystemActive] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [cameraReady, setCameraReady] = useState(false);
    const [wsConnected, setWsConnected] = useState(false);
    const [reconnectAttempts, setReconnectAttempts] = useState(0);

    const { toasts, addToast, removeToast } = useToast();

    // Initialize camera
    useEffect(() => {
        let stream = null;

        const startCamera = async () => {
            try {
                setStatus("Requesting camera access...");
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    setCameraReady(true);
                    setStatus("Camera ready");
                    addToast('Camera connected successfully', 'success');
                }
            } catch (err) {
                console.error("Camera error:", err);
                setStatus("Camera access denied");
                addToast('Please allow camera access to use verification', 'error');
            }
        };

        startCamera();

        return () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    // WebSocket connection with reconnection logic
    useEffect(() => {
        if (!cameraReady) return;

        const connectWebSocket = () => {
            try {
                setStatus("Connecting to server...");
                ws.current = new WebSocket(SOCKET_URL);

                ws.current.onopen = () => {
                    setStatus("Connected - Ready to verify");
                    setSystemActive(true);
                    setWsConnected(true);
                    setReconnectAttempts(0);
                    addToast('Connected to verification server', 'success');
                };

                ws.current.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.error) {
                            console.error(data.error);
                            setStatus("Error processing frame");
                        } else {
                            setResults(data);
                            if (data.length > 0) {
                                const main = data[0];
                                if (main.verified) {
                                    setStatus(`✓ Verified: ${main.name}`);
                                } else {
                                    setStatus("⚠ Unknown User");
                                }
                            } else {
                                setStatus("No face detected");
                            }
                        }
                    } catch (err) {
                        console.error("Message parse error:", err);
                    }
                };

                ws.current.onerror = (error) => {
                    console.error("WebSocket error:", error);
                    setStatus("Connection error");
                    setWsConnected(false);
                };

                ws.current.onclose = () => {
                    setWsConnected(false);
                    setSystemActive(false);
                    setStatus("Disconnected from server");

                    // Attempt reconnection
                    if (reconnectAttempts < 5) {
                        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
                        setStatus(`Reconnecting in ${delay / 1000}s...`);
                        reconnectTimer.current = setTimeout(() => {
                            setReconnectAttempts(prev => prev + 1);
                            connectWebSocket();
                        }, delay);
                    } else {
                        setStatus("Connection failed - Please refresh");
                        addToast('Unable to connect to server. Please check if backend is running.', 'error');
                    }
                };
            } catch (err) {
                console.error("WebSocket creation error:", err);
                setStatus("Failed to connect");
                addToast('Connection error', 'error');
            }
        };

        connectWebSocket();

        return () => {
            if (reconnectTimer.current) {
                clearTimeout(reconnectTimer.current);
            }
            if (ws.current) {
                ws.current.close();
            }
        };
    }, [cameraReady, reconnectAttempts]);

    // Frame capture and send loop
    useEffect(() => {
        if (!systemActive || isPaused) return;

        const interval = setInterval(() => {
            if (videoRef.current && canvasRef.current && ws.current && ws.current.readyState === WebSocket.OPEN) {
                const context = canvasRef.current.getContext('2d');
                context.drawImage(videoRef.current, 0, 0, 640, 480);
                const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.6);
                ws.current.send(dataUrl);
            }
        }, 200); // 5 FPS

        return () => clearInterval(interval);
    }, [systemActive, isPaused]);

    const togglePause = () => {
        setIsPaused(!isPaused);
        addToast(isPaused ? 'Verification resumed' : 'Verification paused', 'info');
    };

    return (
        <>
            <ToastContainer toasts={toasts} removeToast={removeToast} />
            <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2rem', padding: '2rem' }}>
                {/* Header */}
                <div style={{ textAlign: 'center', maxWidth: '800px' }}>
                    <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem', fontWeight: '700' }}>
                        Live <span className="text-gradient">Verification</span>
                    </h1>
                    <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem' }}>
                        Position your face in the camera frame for real-time identity verification
                    </p>
                </div>

                {/* Connection Status Bar */}
                <div className="glass-card" style={{
                    padding: '1rem 1.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1rem',
                    minWidth: '300px'
                }}>
                    {wsConnected ? (
                        <>
                            <Wifi size={20} color="var(--success)" />
                            <span style={{ color: 'var(--success)', fontWeight: '600' }}>Connected</span>
                        </>
                    ) : (
                        <>
                            <WifiOff size={20} color="var(--error)" />
                            <span style={{ color: 'var(--error)', fontWeight: '600' }}>Disconnected</span>
                        </>
                    )}
                    <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
                        <button
                            className="btn-secondary"
                            onClick={togglePause}
                            disabled={!wsConnected}
                            style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }}
                        >
                            {isPaused ? <Play size={16} /> : <Pause size={16} />}
                            {isPaused ? 'Resume' : 'Pause'}
                        </button>
                    </div>
                </div>

                {/* Video Feed */}
                <div className="glass-card hover-lift" style={{ padding: '1rem', position: 'relative', width: 'fit-content' }}>
                    <div style={{
                        position: 'relative',
                        borderRadius: 'var(--radius-md)',
                        overflow: 'hidden',
                        width: '640px',
                        height: '480px',
                        background: '#000',
                        border: `3px solid ${wsConnected ? 'var(--success)' : 'var(--error)'}`
                    }}>
                        <video
                            ref={videoRef}
                            autoPlay
                            muted
                            playsInline
                            width="640"
                            height="480"
                            style={{ transform: 'scaleX(-1)', objectFit: 'cover' }}
                        />
                        <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />

                        {/* Overlay - Status Badge */}
                        <div style={{
                            position: 'absolute',
                            bottom: '20px',
                            left: '50%',
                            transform: 'translateX(-50%)',
                            background: 'rgba(0,0,0,0.8)',
                            padding: '0.75rem 1.5rem',
                            borderRadius: 'var(--radius-full)',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.75rem',
                            backdropFilter: 'blur(8px)',
                            border: '1px solid var(--glass-border)',
                            minWidth: '200px',
                            justifyContent: 'center'
                        }}>
                            {status.includes("Verified") ? (
                                <CheckCircle color="var(--success)" size={20} />
                            ) : status.includes("Unknown") ? (
                                <XCircle color="var(--error)" size={20} />
                            ) : (
                                <AlertTriangle color="var(--warning)" size={20} />
                            )}
                            <span style={{ fontWeight: '600', fontSize: '0.95rem' }}>{status}</span>
                        </div>

                        {/* Paused Overlay */}
                        {isPaused && (
                            <div style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                right: 0,
                                bottom: 0,
                                background: 'rgba(0,0,0,0.7)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                backdropFilter: 'blur(4px)'
                            }}>
                                <div style={{ textAlign: 'center' }}>
                                    <Pause size={48} color="var(--primary)" style={{ marginBottom: '1rem' }} />
                                    <div style={{ fontSize: '1.2rem', fontWeight: '600' }}>Verification Paused</div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Results Section */}
                {results.length > 0 && (
                    <div style={{ width: '100%', maxWidth: '800px' }}>
                        <h3 style={{ fontSize: '1.3rem', fontWeight: '600', marginBottom: '1rem' }}>
                            Detection Results
                        </h3>
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                            gap: '1rem'
                        }}>
                            {results.map((res, i) => (
                                <div key={i} className="glass-card hover-lift" style={{ padding: '1.25rem' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                                        <div>
                                            <h4 style={{ margin: 0, fontSize: '1.2rem', fontWeight: '600' }}>{res.name}</h4>
                                            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                                                Confidence: {res.confidence.toFixed(1)}%
                                            </div>
                                        </div>
                                        <div style={{
                                            padding: '0.35rem 0.85rem',
                                            borderRadius: 'var(--radius-full)',
                                            background: res.verified ? 'var(--success-bg)' : 'var(--error-bg)',
                                            color: res.verified ? 'var(--success)' : 'var(--error)',
                                            fontSize: '0.75rem',
                                            fontWeight: '700',
                                            textTransform: 'uppercase',
                                            letterSpacing: '0.5px'
                                        }}>
                                            {res.verified ? '✓ Match' : '✗ Unknown'}
                                        </div>
                                    </div>
                                    {res.verified && (
                                        <div style={{
                                            marginTop: '0.75rem',
                                            padding: '0.75rem',
                                            background: 'var(--success-bg)',
                                            borderRadius: 'var(--radius-sm)',
                                            fontSize: '0.85rem',
                                            color: 'var(--success)',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.5rem'
                                        }}>
                                            <CheckCircle size={16} />
                                            Identity verified successfully
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Instructions */}
                {results.length === 0 && wsConnected && !isPaused && (
                    <div className="glass-card" style={{ padding: '2rem', maxWidth: '600px', textAlign: 'center' }}>
                        <Camera size={48} color="var(--primary)" style={{ marginBottom: '1rem', opacity: 0.5 }} />
                        <h3 style={{ fontSize: '1.2rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                            Waiting for Face Detection
                        </h3>
                        <p style={{ color: 'var(--text-muted)', lineHeight: '1.6' }}>
                            Position your face clearly in the camera frame. Make sure you're in a well-lit area and looking directly at the camera.
                        </p>
                    </div>
                )}
            </div>
        </>
    );
}

