import React, { useState, useRef, useEffect } from 'react'

const STREAM_URL = 'http://localhost:8000/chat/stream'

const MODES = [
  { id: 'rag', label: 'RAG', icon: '📄', desc: 'Internal docs only' },
  { id: 'hybrid', label: 'Hybrid', icon: '🔀', desc: 'Docs + Web search' },
  { id: 'standalone', label: 'Standalone LLM', icon: '🧠', desc: 'Direct AI answer' },
]

const LLM_MODELS = [
  { id: 'gemini-3.5-flash', name: '3.5 Flash', desc: 'Fastest answers' },
  { id: 'gemini-3.6-flash', name: '3.6 Flash', desc: 'All-around help' },
  { id: 'gemini-3.1-flash-lite', name: '3.1 Flash Lite', desc: 'Lightweight & fast' },
  { id: 'gemini-2.5-flash', name: '2.5 Flash', desc: 'Stable & reliable' },
  { id: 'gemini-2.5-pro', name: '2.5 Pro', desc: 'Advanced reasoning' },
  { id: 'ollama', name: 'Ollama', desc: 'Local Llama 3.1' },
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

function Message({ role, content, sources, streaming }) {
  const isUser = role === 'user'
  return (
    <div className={`message-row ${isUser ? 'from-user' : 'from-assistant'}`}>
      <div className="msg-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="msg-content">
        <div className="message-label">{isUser ? 'You' : 'Assistant'}</div>
        <div className={`message-body${streaming ? ' message-body--streaming' : ''}`}>{content}</div>
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
  const [selectedModel, setSelectedModel] = useState('gemini-3.5-flash')
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [speechSupported, setSpeechSupported] = useState(false)
  const scrollRef = useRef(null)
  const recognitionRef = useRef(null)
  const modelMenuRef = useRef(null)

  // ── Click Outside to Close Menu ──
  useEffect(() => {
    function handleClickOutside(event) {
      if (modelMenuRef.current && !modelMenuRef.current.contains(event.target)) {
        setIsModelMenuOpen(false)
      }
    }
    if (isModelMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isModelMenuOpen])

  // ── Speech Recognition Setup ──
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition

    if (!SpeechRecognition) return // browser doesn't support it

    const recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.interimResults = true
    recognition.continuous = false

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join('')
      setInput(transcript)
    }

    recognition.onend = () => {
      setIsListening(false)
    }

    recognition.onerror = () => {
      setIsListening(false)
    }

    recognitionRef.current = recognition
    setSpeechSupported(true)

    return () => {
      recognition.abort()
    }
  }, [])

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  async function sendQuestion(question) {
    if (!question.trim() || isLoading) return
    setError(null)
    setMessages((prev) => [...prev, { role: 'user', content: question, mode: activeMode }])
    setInput('')
    setIsLoading(true)

    // Add a blank assistant placeholder that we'll stream into
    setMessages((prev) => [...prev, { role: 'assistant', content: '', sources: [], mode: activeMode, streaming: true }])

    // ── Typewriter queue ──────────────────────────────────────────
    // Incoming SSE chunks are pushed here; a fixed-rate interval
    // drains them one character at a time so the UI never jumps.
    const charQueue = []
    const CHAR_DELAY_MS = 18   // ms per character — tweak for speed

    const typingInterval = setInterval(() => {
      if (charQueue.length === 0) return
      const ch = charQueue.shift()
      setMessages((prev) => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        updated[updated.length - 1] = { ...last, content: last.content + ch }
        return updated
      })
    }, CHAR_DELAY_MS)

    // Wait until the queue is drained before applying sources
    function drainQueue() {
      return new Promise((resolve) => {
        const check = setInterval(() => {
          if (charQueue.length === 0) {
            clearInterval(check)
            resolve()
          }
        }, 30)
      })
    }

    try {
      const history = messages
        .map((m) => ({ role: m.role, content: m.content }))
        .slice(-10)

      const res = await fetch(STREAM_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          mode: activeMode,
          history,
          model_name: selectedModel,
        }),
      })

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `Request failed (${res.status})`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let pendingSources = null

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const parts = buffer.split('\n\n')
        buffer = parts.pop()

        for (const part of parts) {
          const line = part.replace(/^data:\s*/, '').trim()
          if (!line) continue
          try {
            const evt = JSON.parse(line)

            if (evt.type === 'token') {
              // Push chars one at a time into the queue — interval drains them
              for (const ch of evt.text) charQueue.push(ch)

            } else if (evt.type === 'replace') {
              // Clear queue and overwrite content immediately
              charQueue.length = 0
              clearInterval(typingInterval)
              setMessages((prev) => {
                const updated = [...prev]
                updated[updated.length - 1] = { ...updated[updated.length - 1], content: evt.text, streaming: false }
                return updated
              })

            } else if (evt.type === 'sources') {
              // Hold sources — apply them after queue drains
              pendingSources = evt

            } else if (evt.type === 'error') {
              // Backend sent a clean error event — clear queue and show error banner
              charQueue.length = 0
              clearInterval(typingInterval)
              setMessages((prev) => prev.filter((m) => !m.streaming))
              setError(evt.message || 'An error occurred. Please try again.')
            }
          } catch {
            // ignore malformed SSE frames
          }
        }
      }

      // Let the typewriter finish before marking done
      await drainQueue()
      clearInterval(typingInterval)

      if (pendingSources) {
        setMessages((prev) => {
          const updated = [...prev]
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            sources: pendingSources.sources,
            mode: pendingSources.mode,
            streaming: false,
          }
          return updated
        })
      }

    } catch (err) {
      clearInterval(typingInterval)
      charQueue.length = 0
      setMessages((prev) => prev.filter((m) => !m.streaming))
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

  function toggleListening() {
    if (!recognitionRef.current) return
    if (isListening) {
      recognitionRef.current.stop()
    } else {
      setInput('')
      recognitionRef.current.start()
      setIsListening(true)
    }
  }

  const activeModeData = MODES.find((m) => m.id === activeMode)

  return (
    <div className="page">
      <header className="page-header">
        <div className="header-avatar">🛡️</div>
        <div className="header-info">
          <h1>Assistant</h1>
          <p className="header-status">
            <span className="status-dot"></span>
            {activeModeData.desc}
          </p>
        </div>
      </header>

      {/* ── Mode Tabs ── */}
      <div className="mode-tabs-bar">
        <div className="tabs-group">
          {MODES.map((mode) => (
            <ModeTab
              key={mode.id}
              mode={mode}
              isActive={activeMode === mode.id}
              onClick={() => setActiveMode(mode.id)}
            />
          ))}
        </div>
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
            <Message key={i} role={m.role} content={m.content} sources={m.sources} streaming={m.streaming} />
          ))}
          {isLoading && !messages.some((m) => m.streaming) && <TypingIndicator />}
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
            isListening
              ? 'Listening…'
              : activeMode === 'rag'
                ? 'Ask about our privacy policy or terms…'
                : activeMode === 'hybrid'
                  ? 'Ask anything (docs + web)…'
                  : 'Ask me anything…'
          }
          disabled={isLoading}
        />

        <div className="model-menu-container" ref={modelMenuRef}>
          <button
            type="button"
            className="model-menu-btn"
            onClick={() => setIsModelMenuOpen(!isModelMenuOpen)}
            title="Select AI Model"
          >
            {LLM_MODELS.find(m => m.id === selectedModel)?.name.replace('gemini-', '')}
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7 10L12 15L17 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </button>

          {isModelMenuOpen && (
            <div className="model-dropdown">
              {LLM_MODELS.map((model) => (
                <button
                  key={model.id}
                  type="button"
                  className={`model-option ${selectedModel === model.id ? 'model-option--active' : ''}`}
                  onClick={() => {
                    setSelectedModel(model.id)
                    setIsModelMenuOpen(false)
                  }}
                >
                  {selectedModel === model.id && (
                    <span className="model-check">
                      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M5 13L9 17L19 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    </span>
                  )}
                  <div className="model-option-text">
                    <span className="model-name">{model.name}</span>
                    <span className="model-desc">{model.desc}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {speechSupported && (
          <button
            type="button"
            className={`mic-btn ${isListening ? 'mic-btn--active' : ''}`}
            onClick={toggleListening}
            disabled={isLoading}
            title={isListening ? 'Stop listening' : 'Voice input'}
          >
            🎙️
          </button>
        )}
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
