import React from 'react'
import { Sun, ShieldAlert, XCircle, Bot } from 'lucide-react'

export default function AnswerPanel({ response }) {
  if (!response) return null

  const { query, answer, success, guardrail_triggered, guardrail_reason } = response

  return (
    <div className="w-full space-y-4">
      {/* Query Echo */}
      {query && (
        <div className="px-2 flex items-start gap-3 opacity-80">
          <div className="w-6 h-6 rounded-full bg-[#3a2824] border border-[#4a3530] flex items-center justify-center shrink-0 mt-0.5">
            <span className="text-xs font-bold text-orange-300">Q</span>
          </div>
          <p className="text-base text-orange-200 font-medium leading-relaxed">
            "{query}"
          </p>
        </div>
      )}

      {/* Answer Card wrapper for gradient border */}
      <div className={`
        relative p-[1px] rounded-2xl transition-all duration-500
        ${guardrail_triggered
          ? 'bg-gradient-to-b from-yellow-500/50 to-transparent shadow-[0_0_30px_rgba(234,179,8,0.15)]'
          : success
          ? 'bg-gradient-to-b from-teal-500/50 to-transparent shadow-[0_0_30px_rgba(20,184,166,0.15)]'
          : 'bg-gradient-to-b from-red-500/50 to-transparent'
        }
      `}>
        {/* Inner Card */}
        <div className="bg-[#1c1311]/95 backdrop-blur-3xl rounded-2xl p-6 sm:p-8 flex items-start gap-4 h-full">
          
          <div className={`
            w-10 h-10 rounded-xl flex items-center justify-center shrink-0 shadow-inner
            ${guardrail_triggered ? 'bg-yellow-500/10 border border-yellow-500/20 text-yellow-500' 
              : success ? 'bg-teal-500/10 border border-teal-500/20 text-teal-400'
              : 'bg-red-500/10 border border-red-500/20 text-red-400'}
          `}>
            {guardrail_triggered ? <ShieldAlert className="w-5 h-5" /> : success ? <Sun className="w-5 h-5" /> : <XCircle className="w-5 h-5" />}
          </div>

          <div className="flex-1 space-y-3 pt-1">
            <div className="flex items-center gap-2 mb-1">
              <Bot className="w-4 h-4 text-orange-200/40" />
              <span className="text-xs font-semibold uppercase tracking-widest text-orange-200/40">AI Response</span>
            </div>
            
            <p className="text-base sm:text-lg text-orange-50 leading-relaxed font-light">
              {answer}
            </p>

            {guardrail_triggered && guardrail_reason && (
              <div className="mt-4 inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-yellow-500"></span>
                <span className="text-xs font-medium text-yellow-400">
                  Guardrail Triggered: {guardrail_reason.replace(/_/g, ' ')}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
