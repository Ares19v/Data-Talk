import React, { useState } from 'react'
import { ChevronDown, ChevronUp, Timer } from 'lucide-react'

function Bar({ label, value, max, color }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-400">
        <span>{label}</span>
        <span className="font-mono">{value?.toFixed(1)}ms</span>
      </div>
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function LatencyPanel({ latency }) {
  const [open, setOpen] = useState(false)
  if (!latency) return null

  const { stt_ms, encode_ms, retrieval_ms, rrf_ms, llm_ms, guardrail_ms, total_pipeline_ms } = latency
  const retrieval_total = (encode_ms || 0) + (retrieval_ms || 0) + (rrf_ms || 0)
  const maxBar = Math.max(stt_ms || 0, llm_ms || 0, 600)

  return (
    <div className="w-full">
      <button
        onClick={() => setOpen((p) => !p)}
        className="
          w-full flex items-center justify-between px-4 py-2.5
          bg-slate-800/40 hover:bg-slate-800/70 border border-slate-700/50
          rounded-xl text-sm text-slate-400 transition
        "
      >
        <span className="flex items-center gap-2">
          <Timer className="w-4 h-4" />
          <span>
            Retrieval{' '}
            <span className={`font-mono font-semibold ${retrieval_total < 200 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {retrieval_total.toFixed(0)}ms
            </span>
            {retrieval_total < 200 && <span className="ml-1 text-emerald-500 text-xs">✓ &lt;200ms</span>}
          </span>
          <span className="text-slate-600">·</span>
          <span>Total <span className="font-mono font-semibold text-slate-300">{total_pipeline_ms?.toFixed(0)}ms</span></span>
        </span>
        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {open && (
        <div className="mt-2 bg-slate-800/40 border border-slate-700/50 rounded-xl p-4 space-y-3">
          {stt_ms != null && (
            <Bar label="🎤 Sarvam STT" value={stt_ms} max={maxBar} color="bg-sky-500" />
          )}
          <Bar label="🔢 Query Encode" value={encode_ms} max={maxBar} color="bg-violet-500" />
          <Bar label="🔍 FAISS Search" value={retrieval_ms} max={maxBar} color="bg-indigo-500" />
          <Bar label="🔀 RRF Fusion" value={rrf_ms} max={maxBar} color="bg-blue-400" />
          <Bar label="🤖 LLM (Groq)" value={llm_ms} max={maxBar} color="bg-emerald-500" />
          <Bar label="🛡️ Guardrails" value={guardrail_ms} max={maxBar} color="bg-amber-500" />

          <div className="pt-2 border-t border-slate-700/50 flex justify-between text-xs">
            <span className="text-slate-500">Retrieval path total</span>
            <span className={`font-mono font-bold ${retrieval_total < 200 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {retrieval_total.toFixed(1)}ms {retrieval_total < 200 ? '✅' : '⚠️'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
