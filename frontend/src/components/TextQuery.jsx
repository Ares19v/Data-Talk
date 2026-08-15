import React, { useState } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { textQuery } from '../api/client'

export default function TextQuery({ onResult, onError, topK }) {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!query.trim() || loading) return
    setLoading(true)
    try {
      const result = await textQuery(query.trim(), topK)
      onResult(result)
    } catch (err) {
      onError(err?.response?.data?.detail || err.message || 'Query failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full flex gap-2">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Or type your question…"
        disabled={loading}
        className="
          flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3
          text-white placeholder-slate-500 text-sm
          focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500
          disabled:opacity-50 transition
        "
      />
      <button
        type="submit"
        disabled={!query.trim() || loading}
        className="
          px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700
          rounded-xl text-white transition flex items-center gap-2
          disabled:cursor-not-allowed text-sm font-medium
        "
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
        <span className="hidden sm:inline">Search</span>
      </button>
    </form>
  )
}
