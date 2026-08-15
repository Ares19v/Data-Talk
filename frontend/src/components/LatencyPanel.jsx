import React, { useState, useEffect } from 'react'
import { ChevronDown, Timer, Zap } from 'lucide-react'

function AnimatedBar({ label, value, max, color }) {
  const [width, setWidth] = useState(0)
  const targetPct = max > 0 ? Math.min((value / max) * 100, 100) : 0

  useEffect(() => {
    // Slight delay so the animation triggers after render
    const t = setTimeout(() => setWidth(targetPct), 100)
    return () => clearTimeout(t)
  }, [targetPct])

  return (
    <div className="space-y-1.5 group">
      <div className="flex justify-between text-xs">
        <span className="text-orange-200/60 font-medium">{label}</span>
        <span className="font-mono text-orange-200/80">{value?.toFixed(1)}ms</span>
      </div>
      <div className="h-2 bg-black/40 rounded-full overflow-hidden shadow-inner border border-white/5">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out relative overflow-hidden ${color}`}
          style={{ width: `${width}%` }}
        >
          <div className="absolute inset-0 bg-white/20 w-full h-full transform -skew-x-12 -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]"></div>
        </div>
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

  const isFast = retrieval_total < 200

  return (
    <div className="w-full bg-[#1c1311]/80 backdrop-blur-xl border border-orange-900/30 rounded-2xl overflow-hidden transition-all duration-300">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-5 hover:bg-orange-500/[0.02] transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center">
            <Timer className="w-4 h-4 text-teal-400" />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-orange-100">Retrieval</span>
            <span className={`font-mono font-bold ${isFast ? 'text-teal-400' : 'text-yellow-400'}`}>
              {retrieval_total.toFixed(0)}ms
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-orange-200/50 font-mono hidden sm:inline">Total {total_pipeline_ms?.toFixed(0)}ms</span>
          <ChevronDown className={`w-5 h-5 text-orange-200/50 transition-transform duration-300 ${open ? 'rotate-180' : ''}`} />
        </div>
      </button>

      <div className={`grid transition-all duration-300 ease-in-out ${open ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
        <div className="overflow-hidden">
          <div className="p-5 pt-0 space-y-4">
            {stt_ms != null && (
              <AnimatedBar label="🎤 Sarvam STT (Network)" value={stt_ms} max={maxBar} color="bg-gradient-to-r from-orange-400 to-red-500" />
            )}
            <AnimatedBar label="🔢 Query Encode" value={encode_ms} max={maxBar} color="bg-gradient-to-r from-red-500 to-rose-600" />
            <AnimatedBar label="🔍 FAISS Search" value={retrieval_ms} max={maxBar} color="bg-gradient-to-r from-yellow-400 to-orange-500" />
            <AnimatedBar label="🔀 RRF Fusion" value={rrf_ms} max={maxBar} color="bg-gradient-to-r from-[#d89656] to-[#c45a41]" />
            <AnimatedBar label="🤖 LLM Gen (Groq)" value={llm_ms} max={maxBar} color="bg-gradient-to-r from-teal-400 to-[#45867c]" />
            <AnimatedBar label="🛡️ Guardrails" value={guardrail_ms} max={maxBar} color="bg-gradient-to-r from-yellow-500 to-amber-600" />

            <div className="mt-6 pt-4 border-t border-[#4a3530] flex items-center justify-between">
              <span className="text-xs font-medium uppercase tracking-wider text-orange-200/50">Retrieval Target</span>
              <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border ${isFast ? 'bg-teal-500/10 border-teal-500/20 text-teal-400' : 'bg-yellow-500/10 border-yellow-500/20 text-yellow-400'}`}>
                {isFast && <Zap className="w-3.5 h-3.5 fill-current" />}
                <span className="text-xs font-bold font-mono">{retrieval_total.toFixed(1)}ms / 200ms</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
