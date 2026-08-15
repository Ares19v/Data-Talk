import React, { useState } from 'react'
import { ChevronDown, ChevronUp, BookOpen, Star } from 'lucide-react'

const STRATEGY_COLORS = {
  fixed_size:          'bg-blue-900/40 text-blue-300 border-blue-800/40',
  semantic_sentence:   'bg-purple-900/40 text-purple-300 border-purple-800/40',
  passage_aware:       'bg-emerald-900/40 text-emerald-300 border-emerald-800/40',
  hierarchical_parent: 'bg-amber-900/40 text-amber-300 border-amber-800/40',
  hierarchical_child:  'bg-orange-900/40 text-orange-300 border-orange-800/40',
  combined:            'bg-slate-700/40 text-slate-300 border-slate-600/40',
}

function strategyLabel(s) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export default function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources || sources.length === 0) return null

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
          <BookOpen className="w-4 h-4" />
          {sources.length} source passage{sources.length !== 1 ? 's' : ''} retrieved
        </span>
        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          {sources.map((src, i) => {
            const colorClass = STRATEGY_COLORS[src.strategy] || STRATEGY_COLORS.combined
            return (
              <div
                key={i}
                className={`rounded-xl p-4 border text-xs space-y-2 ${colorClass}`}
              >
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold opacity-60">[{i + 1}]</span>
                    <span className="font-medium">{strategyLabel(src.strategy)}</span>
                    {src.is_selected === 1 && (
                      <span className="flex items-center gap-0.5 text-yellow-400">
                        <Star className="w-3 h-3 fill-current" />
                        <span className="text-yellow-400/80">ground truth</span>
                      </span>
                    )}
                  </div>
                  <div className="flex gap-3 opacity-70 font-mono text-[10px]">
                    <span>RRF {src.rrf_score.toFixed(4)}</span>
                    <span>cos {src.faiss_score.toFixed(3)}</span>
                  </div>
                </div>
                <p className="leading-relaxed opacity-90 line-clamp-4">{src.text}</p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
