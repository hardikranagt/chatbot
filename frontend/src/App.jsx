import React, { useState, useRef, useEffect } from 'react'

const API_URL = 'http://localhost:8000/chat'

const MODES = [
  { id: 'rag', label: 'RAG', icon: '📄', desc: 'Internal docs only' },
  { id: 'hybrid', label: 'Hybrid', icon: '🔀', desc: 'Docs + Web search' },
  { id: 'standalone', label: 'Standalone LLM', icon: '🧠', desc: 'Direct AI answer' },
]

const STARTER_QUESTIONS = [
  'How long do you keep my personal data?',
  'Can I request my account be deleted?',
  'What happens if I violate the terms?',
]

function SourceTag({ source }) {
  if (source.type === 'web') {
    return (
      <a
        href={source.link}
        target="_blank"
        rel="noopener noreferrer"
        className="source-tag source-tag--web"
        title={source.document}
      >
        🌐 {source.document?.length > 35
          ? source.document.substring(0, 35) + '…'
          : source.document}
      </a>
    )
  }

  const label =
    source.document === 'privacy_policy' ? 'Privacy Policy' : 'Terms & Conditions'
  return (
    <span className="source-tag">
      📄 {label} · p.{source.page}
    </span>
  )
}

function Message({ role, content, sources }) {
  const isUser = role === 'user'
  return (
    <div className={`message-row ${isUser ? 'from-user' : 'from-assistant'}`}>
      <div className="msg-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="msg-content">
        <div className="message-label">{isUser ? 'You' : 'Assistant'}</div>
        <div className="message-body">{content}</div>
        {sources && sources.length > 0 && (
          <div className="sources-list">
            {sources.map((s, i) => (
              <SourceTag key={i} source={s} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message-row from-assistant">
      <div className="msg-avatar">🤖</div>
      <div className="msg-content">
        <div className="message-label">Assistant</div>
        <div className="message-body typing">
          <div className="typing-dots">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    </div>
  )
}

function ModeTab({ mode, isActive, onClick }) {
  return (
    <button
      type="button"
      className={`mode-tab ${isActive ? 'mode-tab--active' : ''}`}
      onClick={onClick}
      title={mode.desc}
    >
      <span className="mode-tab-icon">{mode.icon}</span>
      <span className="mode-tab-label">{mode.label}</span>
    </button>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeMode, setActiveMode] = useState('rag')
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  async function sendQuestion(question) {
    if (!question.trim() || isLoading) return

    setError(null)
    setMessages((prev) => [...prev, { role: 'user', content: question, mode: activeMode }])
    setInput('')
    setIsLoading(true)

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, mode: activeMode }),
      })

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `Request failed (${res.status})`)
      }

      const data = await res.json()
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
          mode: data.mode,
        },
      ])
    } catch (err) {
      setError(
        err.message ||
        'Could not reach the assistant. Is the backend running on port 8000?'
      )
    } finally {
      setIsLoading(false)
    }
  }

  function handleSubmit(e) {
    e.preventDefault()
    sendQuestion(input)
  }

  const activeModeData = MODES.find((m) => m.id === activeMode)

  return (
    <div className="page">
      <header className="page-header">
        <div className="header-avatar">🛡️</div>
        <div className="header-info">
          <h1>Policy Assistant</h1>
          <p className="header-status">
            <span className="status-dot"></span>
            {activeModeData.desc}
          </p>
        </div>
      </header>

      {/* ── Mode Tabs ── */}
      <div className="mode-tabs-bar">
        {MODES.map((mode) => (
          <ModeTab
            key={mode.id}
            mode={mode}
            isActive={activeMode === mode.id}
            onClick={() => setActiveMode(mode.id)}
          />
        ))}
      </div>

      <main className="chat-panel">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <h2>How can I help?</h2>
            <p>
              {activeMode === 'rag' && 'Ask anything about our privacy practices or terms of service.'}
              {activeMode === 'hybrid' && "Ask anything — I'll search our docs and the web."}
              {activeMode === 'standalone' && "Ask me anything — I'll answer from my training data."}
            </p>
            <div className="starter-list">
              {STARTER_QUESTIONS.map((q) => (
                <button
                  key={q}
                  type="button"
                  className="starter-chip"
                  onClick={() => sendQuestion(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="message-list">
          {messages.map((m, i) => (
            <Message key={i} role={m.role} content={m.content} sources={m.sources} />
          ))}
          {isLoading && <TypingIndicator />}
          {error && <div className="error-banner">{error}</div>}
          <div ref={scrollRef} />
        </div>
      </main>

      <form className="input-bar" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            activeMode === 'rag'
              ? 'Ask about our privacy policy or terms…'
              : activeMode === 'hybrid'
                ? 'Ask anything (docs + web)…'
                : 'Ask me anything…'
          }
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
