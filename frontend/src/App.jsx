import React, { useState, useEffect } from 'react'
import { Activity, AlertCircle, Wifi, WifiOff } from 'lucide-react'
import { healthCheck } from './api/client'
import VoiceRecorder from './components/VoiceRecorder'
import TextQuery from './components/TextQuery'
import AnswerPanel from './components/AnswerPanel'
import SourcesPanel from './components/SourcesPanel'
import LatencyPanel from './components/LatencyPanel'

export default function App() {
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)
  const [topK, setTopK] = useState(5)

  // Poll health on mount
  useEffect(() => {
    healthCheck()
      .then(setHealth)
      .catch(() => setHealth({ status: 'unreachable' }))
  }, [])

  const handleResult = (data) => {
    setError(null)
    setResponse(data)
  }

  const handleError = (msg) => {
    setError(msg)
    setResponse(null)
  }

  const indexLoaded = health?.index_loaded
  const totalVectors = health?.total_vectors

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight">
              🎙️ Voice RAG
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">
              ai4bharat/MSMARCO-XI · HH Goa 2026
            </p>
          </div>

          {/* Health badge */}
          <div className="flex items-center gap-2">
            {health ? (
              indexLoaded ? (
                <span className="flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-950/50 border border-emerald-800/40 px-2.5 py-1 rounded-full">
                  <Wifi className="w-3 h-3" />
                  {totalVectors?.toLocaleString()} vectors
                </span>
              ) : (
                <span className="flex items-center gap-1.5 text-xs text-amber-400 bg-amber-950/50 border border-amber-800/40 px-2.5 py-1 rounded-full">
                  <AlertCircle className="w-3 h-3" />
                  Index not built
                </span>
              )
            ) : (
              <span className="flex items-center gap-1.5 text-xs text-slate-500 bg-slate-800/50 border border-slate-700/40 px-2.5 py-1 rounded-full">
                <WifiOff className="w-3 h-3" />
                Connecting…
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-3xl mx-auto px-6 py-10 space-y-8">

        {/* Index not built warning */}
        {health && !indexLoaded && (
          <div className="bg-amber-950/30 border border-amber-800/40 rounded-xl px-4 py-3 text-sm text-amber-300">
            ⚠️ FAISS index not built yet. Run{' '}
            <code className="font-mono bg-amber-950/50 px-1.5 py-0.5 rounded">
              python scripts/build_index.py
            </code>{' '}
            to index the dataset, then restart the backend.
          </div>
        )}

        {/* Voice recorder */}
        <div className="flex flex-col items-center gap-2">
          <p className="text-sm text-slate-500 font-medium uppercase tracking-widest mb-2">
            Speak your question
          </p>
          <VoiceRecorder onResult={handleResult} onError={handleError} topK={topK} />
        </div>

        {/* Divider */}
        <div className="flex items-center gap-3">
          <div className="flex-1 h-px bg-slate-800" />
          <span className="text-xs text-slate-600">or</span>
          <div className="flex-1 h-px bg-slate-800" />
        </div>

        {/* Text query */}
        <TextQuery onResult={handleResult} onError={handleError} topK={topK} />

        {/* Top-K slider */}
        <div className="flex items-center gap-3">
          <label className="text-xs text-slate-500 whitespace-nowrap">
            Context chunks (top-k):
          </label>
          <input
            type="range"
            min={1} max={10} step={1}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="flex-1 accent-indigo-500"
          />
          <span className="text-xs font-mono text-slate-400 w-4">{topK}</span>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-950/30 border border-red-800/40 rounded-xl px-4 py-3 text-sm text-red-300 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            {error}
          </div>
        )}

        {/* Results */}
        {response && (
          <div className="space-y-3">
            <AnswerPanel response={response} />
            <SourcesPanel sources={response.sources} />
            <LatencyPanel latency={response.latency} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 px-6 py-4 mt-10">
        <div className="max-w-3xl mx-auto flex items-center justify-between text-xs text-slate-600">
          <span>4 chunking strategies · RRF fusion · FAISS IVFFlat</span>
          <span className="flex items-center gap-1">
            <Activity className="w-3 h-3" />
            Sarvam STT · Groq LLaMA 3.1
          </span>
        </div>
      </footer>
    </div>
  )
}
