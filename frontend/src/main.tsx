import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import MobileApp from './MobileApp'
import './index.css'

// Which UI to mount. The desktop tool is built around a pan/zoom canvas, hover tooltips
// and side-by-side layers, none of which work on a phone; the mobile view is a separate
// touch-first surface over the same API. `?ui=mobile|desktop` overrides the width test so
// either can be checked from either device.
function pickUI(): 'mobile' | 'desktop' {
  const forced = new URLSearchParams(window.location.search).get('ui')
  if (forced === 'mobile' || forced === 'desktop') return forced
  const coarse = window.matchMedia('(pointer: coarse)').matches
  return window.innerWidth <= 820 && coarse ? 'mobile' : 'desktop'
}

const Root = pickUI() === 'mobile' ? MobileApp : App

// A render error in React 18 unmounts the WHOLE tree, leaving the page background and
// nothing else — which is indistinguishable from a hang and impossible to report from a
// phone. Show the error instead.
class ErrorBoundary extends React.Component<{ children: React.ReactNode },
                                            { err: Error | null; info: string }> {
  constructor(p: any) { super(p); this.state = { err: null, info: '' } }
  static getDerivedStateFromError(err: Error) { return { err, info: '' } }
  componentDidCatch(err: Error, info: React.ErrorInfo) {
    this.setState({ err, info: info.componentStack ?? '' })
    console.error('[RidgeExplorer] render error', err, info)
  }
  render() {
    if (!this.state.err) return this.props.children
    return (
      <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif', color: '#eee',
                    background: '#12121e', minHeight: '100vh' }}>
        <h2 style={{ color: '#e94560', fontSize: 16 }}>The interface hit an error</h2>
        <p style={{ fontSize: 13, color: '#bbc' }}>
          The run itself is unaffected — this is a display fault. Reloading is safe.
        </p>
        <pre style={{ fontSize: 11, background: '#0e0e18', padding: 10, borderRadius: 6,
                      whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#f4a' }}>
          {String(this.state.err?.stack || this.state.err)}
        </pre>
        <pre style={{ fontSize: 10, background: '#0e0e18', padding: 10, borderRadius: 6,
                      whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#89a',
                      maxHeight: 220, overflow: 'auto' }}>
          {this.state.info}
        </pre>
        <button onClick={() => window.location.reload()}
                style={{ minHeight: 44, padding: '0 18px', fontSize: 14, borderRadius: 8,
                         border: 'none', background: '#4ecca3', color: '#fff' }}>
          reload
        </button>
      </div>
    )
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <Root />
    </ErrorBoundary>
  </React.StrictMode>
)
