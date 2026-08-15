import React, { useState } from 'react'
import { ChevronDown, BookOpen, Star } from 'lucide-react'

const STRATEGY_STYLES = {
  fixed_size:          { bg: 'bg-teal-500/10', border: 'border-teal-500/20', text: 'text-teal-400' },
  semantic_sentence:   { bg: 'bg-orange-500/10', border: 'border-orange-500/20', text: 'text-orange-400' },
  passage_aware:       { bg: 'bg-yellow-500/10', border: 'border-yellow-500/20', text: 'text-yellow-400' },
  hierarchical_parent: { bg: 'bg-red-500/10', border: 'border-red-500/20', text: 'text-red-400' },
  hierarchical_child:  { bg: 'bg-[#d89656]/10', border: 'border-[#d89656]/20', text: 'text-[#d89656]' },
  combined:            { bg: 'bg-[#4b3631]/50', border: 'border-[#4b3631]', text: 'text-orange-200' },
}

function formatStrategy(s) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export default function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false)
  if (!sources || sources.length === 0) return null

  return (
    <div className="w-full bg-[#1c1311]/80 backdrop-blur-xl border border-orange-900/30 rounded-2xl overflow-hidden transition-all duration-300">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-5 hover:bg-orange-500/[0.02] transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-orange-500/10 border border-orange-500/20 flex items-center justify-center">
            <BookOpen className="w-4 h-4 text-orange-400" />
          </div>
          <span className="font-medium text-orange-100">
            {sources.length} Context Passage{sources.length !== 1 ? 's' : ''}
          </span>
        </div>
        <ChevronDown className={`w-5 h-5 text-orange-200/50 transition-transform duration-300 ${open ? 'rotate-180' : ''}`} />
      </button>

      <div className={`grid transition-all duration-300 ease-in-out ${open ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
        <div className="overflow-hidden">
          <div className="p-5 pt-0 space-y-3">
            {sources.map((src, i) => {
              const style = STRATEGY_STYLES[src.strategy] || STRATEGY_STYLES.combined
              return (
                <div key={i} className={`rounded-xl p-4 border ${style.bg} ${style.border}`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold opacity-50 text-orange-200">#{i + 1}</span>
                      <span className={`text-xs font-semibold tracking-wide uppercase ${style.text}`}>
                        {formatStrategy(src.strategy)}
                      </span>
                      {src.is_selected === 1 && (
                        <span className="flex items-center gap-1 bg-yellow-500/10 border border-yellow-500/20 text-yellow-500 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider">
                          <Star className="w-3 h-3 fill-current" /> Truth
                        </span>
                      )}
                    </div>
                    <div className="flex gap-2 font-mono text-[10px] text-orange-200/50 bg-black/40 px-2 py-1 rounded">
                      <span>RRF {src.rrf_score.toFixed(3)}</span>
                      <span className="opacity-50">|</span>
                      <span>COS {src.faiss_score.toFixed(2)}</span>
                    </div>
                  </div>
                  <p className="text-sm text-orange-200/80 leading-relaxed font-light">{src.text}</p>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
