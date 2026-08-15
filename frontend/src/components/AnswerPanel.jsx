import React from 'react'
import { CheckCircle, XCircle, ShieldAlert, Sparkles } from 'lucide-react'

export default function AnswerPanel({ response }) {
  if (!response) return null

  const { query, answer, success, guardrail_triggered, guardrail_reason, latency } = response

  return (
    <div className="w-full space-y-4 animate-fade-in">
      {/* Query echo */}
      {query && (
        <div className="text-xs text-slate-500 px-1">
          <span className="font-medium text-slate-400">Q:</span> {query}
        </div>
      )}

      {/* Answer card */}
      <div className={`
        rounded-2xl p-5 border
        ${guardrail_triggered
          ? 'bg-amber-950/30 border-amber-800/40'
          : success
          ? 'bg-slate-800/60 border-slate-700/50'
          : 'bg-red-950/30 border-red-800/40'
        }
      `}>
        <div className="flex items-start gap-3">
          <div className="mt-0.5">
            {guardrail_triggered ? (
              <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0" />
            ) : success ? (
              <Sparkles className="w-5 h-5 text-indigo-400 shrink-0" />
            ) : (
              <XCircle className="w-5 h-5 text-red-400 shrink-0" />
            )}
          </div>
          <p className="text-sm leading-relaxed text-slate-100">{answer}</p>
        </div>

        {guardrail_triggered && guardrail_reason && (
          <p className="mt-2 ml-8 text-xs text-amber-500/70">
            Guardrail: {guardrail_reason.replace(/_/g, ' ')}
          </p>
        )}
      </div>
    </div>
  )
}
