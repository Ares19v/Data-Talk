import React, { useState } from 'react'
import { ArrowRight, Loader2, Search } from 'lucide-react'
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
    <form onSubmit={handleSubmit} className="w-full relative group">
      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
        <Search className="h-5 w-5 text-white/40 group-focus-within:text-white transition-colors" />
      </div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Type a manual query instead..."
        disabled={loading}
        className="
          w-full pl-12 pr-16 py-4 
          bg-[#231815]/50 backdrop-blur-md 
          border border-white/10 rounded-2xl
          text-white placeholder-white/40 text-base
          shadow-inner
          focus:outline-none focus:border-white/30 focus:bg-[#231815]/80 focus:ring-4 focus:ring-white/10
          disabled:opacity-50 transition-all duration-300
        "
      />
      <div className="absolute inset-y-0 right-2 flex items-center">
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="
            p-2 bg-orange-500 hover:bg-orange-400 disabled:bg-white/5 disabled:text-white/30
            text-white rounded-xl transition-all duration-300 
            disabled:cursor-not-allowed flex items-center justify-center
            hover:shadow-[0_0_15px_rgba(249,115,22,0.4)]
          "
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <ArrowRight className="w-5 h-5" />}
        </button>
      </div>
    </form>
  )
}
