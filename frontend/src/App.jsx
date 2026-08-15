import React, { useState, useEffect } from 'react'
import { Activity, AlertCircle, Wifi, WifiOff, TreePalm, Database, Info, X } from 'lucide-react'
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
  const [showTopK, setShowTopK] = useState(false)
  const [showInfo, setShowInfo] = useState(false)

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
    <div className="min-h-screen relative selection:bg-teal-500/30">
      
      {/* Scenic SVG Background (Mountains, Sun, Seagulls) */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        {/* Sun with white crescent shadow to match poster */}
        <div className="absolute top-[12vh] right-[15vw] md:right-[25vw] w-28 h-28 sm:w-40 sm:h-40 rounded-full bg-[#d35529] shadow-[inset_-6px_-6px_0_0_rgba(255,255,255,0.9)]"></div>
        
        {/* Bird Silhouettes (Non-interactive) */}
        <div className="absolute top-[12vh] left-[8vw] md:left-[12vw] pointer-events-none opacity-90">
          <svg className="w-16 h-8 absolute top-0 left-0 transform -rotate-6" viewBox="0 0 100 40" fill="#2b1b1a">
            <path d="M0,20 Q25,-10 50,20 Q75,-10 100,20 Q75,10 50,15 Q25,10 0,20 Z" />
          </svg>
          <svg className="w-12 h-6 absolute top-[-20px] left-[70px] transform -rotate-12" viewBox="0 0 100 40" fill="#2b1b1a">
            <path d="M0,20 Q25,-10 50,20 Q75,-10 100,20 Q75,10 50,15 Q25,10 0,20 Z" />
          </svg>
          <svg className="w-8 h-4 absolute top-[15px] left-[110px] transform rotate-3" viewBox="0 0 100 40" fill="#2b1b1a">
            <path d="M0,20 Q25,-10 50,20 Q75,-10 100,20 Q75,10 50,15 Q25,10 0,20 Z" />
          </svg>
          <svg className="w-6 h-3 absolute top-[-5px] left-[140px] transform -rotate-6" viewBox="0 0 100 40" fill="#2b1b1a">
            <path d="M0,20 Q25,-10 50,20 Q75,-10 100,20 Q75,10 50,15 Q25,10 0,20 Z" />
          </svg>
        </div>

        {/* Burgundy Mountains sitting exactly on the 45vh horizon line */}
        <svg className="absolute top-[25vh] w-full h-[20vh]" viewBox="0 0 100 100" preserveAspectRatio="none">
          <polygon fill="#692931" points="0,100 12,65 25,80 45,20 52,45 62,35 80,65 100,50 100,100" />
        </svg>

        {/* Goa Coconut Trees - Rooted in the sand, rising past the horizon */}
        
        {/* Left Tree */}
        <div className="pointer-events-auto absolute bottom-0 left-[2vw] md:left-[5vw] h-[65vh] opacity-95 transition-all duration-700 ease-out hover:scale-[1.04] hover:-rotate-2 origin-bottom cursor-pointer drop-shadow-none hover:drop-shadow-[0_0_20px_rgba(229,169,95,0.4)]">
          <svg className="h-full w-auto" viewBox="0 0 100 200" fill="#2b1b1a">
            {/* Trunk */}
            <path d="M46,200 Q40,100 50,40 Q52,100 54,200 Z" />
            {/* Drooping Fronds */}
            <path d="M50,40 Q10,10 0,50 Q25,40 50,40 Z" />
            <path d="M50,40 Q90,10 100,50 Q75,40 50,40 Z" />
            <path d="M50,40 Q20,30 15,80 Q35,50 50,40 Z" />
            <path d="M50,40 Q80,30 85,80 Q65,50 50,40 Z" />
            {/* Top Fronds */}
            <path d="M50,40 Q40,15 35,20 Q45,30 50,40 Z" />
            <path d="M50,40 Q60,15 65,20 Q55,30 50,40 Z" />
            {/* Coconuts */}
            <circle cx="45" cy="44" r="3" />
            <circle cx="55" cy="44" r="3" />
            <circle cx="50" cy="48" r="3.5" />
          </svg>
        </div>

        {/* Right Tree */}
        <div className="pointer-events-auto absolute -bottom-[5vh] right-[2vw] md:right-[5vw] h-[75vh] opacity-95 transition-all duration-700 ease-out hover:scale-[1.04] hover:rotate-2 origin-bottom cursor-pointer drop-shadow-none hover:drop-shadow-[0_0_20px_rgba(229,169,95,0.4)]">
          <svg className="h-full w-auto transform -scale-x-100" viewBox="0 0 100 200" fill="#2b1b1a">
            {/* Trunk */}
            <path d="M46,200 Q30,100 50,40 Q52,100 54,200 Z" />
            {/* Drooping Fronds */}
            <path d="M50,40 Q10,10 0,50 Q25,40 50,40 Z" />
            <path d="M50,40 Q90,10 100,50 Q75,40 50,40 Z" />
            <path d="M50,40 Q20,30 15,80 Q35,50 50,40 Z" />
            <path d="M50,40 Q80,30 85,80 Q65,50 50,40 Z" />
            {/* Top Fronds */}
            <path d="M50,40 Q40,15 35,20 Q45,30 50,40 Z" />
            <path d="M50,40 Q60,15 65,20 Q55,30 50,40 Z" />
            {/* Coconuts */}
            <circle cx="45" cy="44" r="3" />
            <circle cx="55" cy="44" r="3" />
            <circle cx="50" cy="48" r="3.5" />
          </svg>
        </div>
      </div>

      {/* Header */}
      <header className="sticky top-0 z-50 bg-[#e5a95f]/30 backdrop-blur-md border-b border-[#c45a41]/20">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-sm bg-[#2b1b1a] flex items-center justify-center shadow-lg shadow-[#2b1b1a]/20">
              <TreePalm className="w-6 h-6 text-[#e5a95f]" />
            </div>
            <div>
              <h1 className="text-3xl font-['Alfa_Slab_One'] tracking-wider text-[#2b1b1a] drop-shadow-[1px_1px_0_rgba(255,255,255,0.8)] leading-none">
                DATA TALK
              </h1>
            </div>
          </div>

          <div className="flex items-center">
            {health ? (
              indexLoaded ? null : (
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-orange-500/20 border border-orange-500/30">
                  <AlertCircle className="w-3.5 h-3.5 text-orange-400" />
                  <span className="text-xs font-medium text-orange-400">Index missing</span>
                </div>
              )
            ) : (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#3a2824]/50 border border-[#4a3530]">
                <WifiOff className="w-3.5 h-3.5 text-orange-200/50" />
                <span className="text-xs font-medium text-orange-200/50">Connecting...</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-4xl mx-auto px-6 py-12 min-h-[100vh] flex flex-col justify-center relative z-10 space-y-10">
        
        {/* Main interactive card (Silhouette style) */}
        <div className="relative animate-slide-up">
          <div className="absolute -inset-0.5 bg-gradient-to-b from-orange-500/30 to-teal-500/20 rounded-[32px] blur-xl opacity-60"></div>
          <div className="relative bg-[#1c1311]/90 backdrop-blur-2xl border border-white/10 rounded-[32px] p-8 sm:p-12 shadow-[0_20px_50px_rgba(0,0,0,0.5)] flex flex-col items-center">
            
            <div className="text-center mb-10 space-y-3">
              <h2 className="text-3xl font-semibold tracking-tight text-orange-50">Ask anything.</h2>
              <p className="text-orange-200/60 text-sm max-w-md mx-auto leading-relaxed">
                Speak or type your question. The system will retrieve knowledge from the MSMARCO-XI dataset and generate a grounded answer.
              </p>
            </div>

            <VoiceRecorder onResult={handleResult} onError={handleError} topK={topK} />

            <div className="w-full flex items-center gap-4 my-10 opacity-60">
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-orange-900/50 to-transparent" />
              <span className="text-xs font-medium text-orange-200/50 uppercase tracking-widest">OR</span>
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-orange-900/50 to-transparent" />
            </div>

            <TextQuery onResult={handleResult} onError={handleError} topK={topK} />

            <div className="w-full mt-6 text-center">
              <span className="text-xs text-orange-100/30 font-medium italic tracking-wide">
                Try asking: "What is a corporation?"
              </span>
            </div>

            <div className="w-full mt-8 pt-6 border-t border-white/5 flex justify-end">
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowTopK(!showTopK)}
                  className="text-xs font-medium text-orange-200/60 hover:text-orange-200 uppercase tracking-wider flex items-center gap-2 transition-colors cursor-pointer"
                >
                  <Database className="w-4 h-4 text-orange-400" />
                  Context Depth (Top-K): {topK}
                </button>
                
                {showTopK && (
                  <div className="absolute right-0 bottom-full mb-3 w-64 bg-[#1c1311] border border-orange-500/20 rounded-xl p-4 shadow-2xl shadow-black/50 animate-fade-in z-50">
                    <div className="flex items-center gap-4">
                      <input
                        type="range"
                        min={1} max={10} step={1}
                        value={topK}
                        onChange={(e) => setTopK(Number(e.target.value))}
                        className="w-full h-1.5 bg-[#3a2824] rounded-lg appearance-none cursor-pointer accent-orange-500 hover:accent-orange-400 transition"
                      />
                      <span className="text-sm font-mono font-bold text-orange-300 bg-orange-500/20 border border-orange-500/30 px-2 py-1 rounded-md">
                        {topK}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Error State */}
        {error && (
          <div className="animate-fade-in bg-red-950/80 border border-red-500/30 backdrop-blur-md rounded-2xl p-5 flex items-start gap-3 shadow-lg shadow-red-900/20">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <p className="text-sm text-red-200 leading-relaxed">{error}</p>
          </div>
        )}

        {/* Results Area */}
        {response && (
          <div className="space-y-6 animate-slide-up" style={{ animationDelay: '0.1s' }}>
            <AnswerPanel response={response} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <SourcesPanel sources={response.sources} />
              <LatencyPanel latency={response.latency} />
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 bg-[#1c1311]/90 backdrop-blur-lg mt-12 py-8 relative z-10">
        <div className="max-w-4xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-orange-200/40">
          <p>Built for HH Goa 2026 Shortlisting Task 2</p>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5"><Activity className="w-3.5 h-3.5 text-orange-200/30" /> Sarvam STT</span>
              <span className="flex items-center gap-1.5"><Activity className="w-3.5 h-3.5 text-orange-200/30" /> Groq LLaMA 3.1</span>
            </div>
            <div className="w-px h-4 bg-white/10 hidden sm:block"></div>
            <button 
              onClick={() => setShowInfo(true)}
              className="flex items-center gap-2 hover:text-orange-200 transition-colors group cursor-pointer"
            >
              <Info className="w-4 h-4 group-hover:scale-110 transition-transform" />
              <span className="hidden sm:inline font-bold uppercase tracking-widest text-[10px]">Architecture</span>
            </button>
          </div>
        </div>
      </footer>

      {/* Info Modal Overlay */}
      {showInfo && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6 animate-fade-in">
          {/* Backdrop - dims everything else */}
          <div 
            className="absolute inset-0 bg-black/70 backdrop-blur-sm cursor-pointer"
            onClick={() => setShowInfo(false)}
          ></div>
          
          {/* Glass Modal Content */}
          <div className="relative bg-[#1c1311]/95 backdrop-blur-3xl border border-orange-500/20 rounded-[32px] p-8 md:p-10 shadow-[0_20px_60px_rgba(0,0,0,0.8)] max-w-2xl w-full z-10 overflow-hidden transform transition-all animate-slide-up">
            <button 
              onClick={() => setShowInfo(false)}
              className="absolute top-6 right-6 p-2 text-orange-200/50 hover:text-white transition-colors bg-white/5 rounded-full hover:bg-white/10 cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
            
            <h2 className="text-2xl font-bold text-orange-50 mb-8 flex items-center gap-3">
              <Info className="w-6 h-6 text-orange-400" />
              Project Architecture
            </h2>
            
            <div className="space-y-6 text-sm text-orange-200/80 leading-relaxed max-h-[60vh] overflow-y-auto pr-4">
              
              <section>
                <h3 className="text-orange-300 font-bold uppercase tracking-wider text-[11px] mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-orange-400"></span> Speech-to-Text
                </h3>
                <p>Audio is captured natively in the browser and streamed to the <strong>Sarvam AI STT API</strong>, ensuring ultra-fast and highly accurate transcription of spoken Indian/English queries.</p>
              </section>

              <section>
                <h3 className="text-teal-400 font-bold uppercase tracking-wider text-[11px] mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-teal-400"></span> Vector Retrieval & Dataset
                </h3>
                <p>The knowledge base is built on the <strong>AI4Bharat MSMARCO-XI</strong> dataset. Over 53,000 document chunks were embedded locally using the <code className="text-teal-200 bg-teal-500/10 px-1.5 py-0.5 rounded font-mono text-[10px]">all-MiniLM-L6-v2</code> model and indexed into a highly optimized <strong>FAISS</strong> database for millisecond-level similarity search.</p>
              </section>

              <section>
                <h3 className="text-yellow-400 font-bold uppercase tracking-wider text-[11px] mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-yellow-400"></span> Advanced RAG (RRF)
                </h3>
                <p>Instead of relying on a single chunking technique, the backend performs parallel searches across four distinct chunking strategies (Semantic, Fixed Size, Hierarchical Parent, and Child). Results are merged and scored using <strong>Reciprocal Rank Fusion (RRF)</strong> to mathematically guarantee the highest quality context retrieval.</p>
              </section>

              <section>
                <h3 className="text-[#e5a95f] font-bold uppercase tracking-wider text-[11px] mb-2 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#e5a95f]"></span> Generation & Guardrails
                </h3>
                <p>Retrieved passages are passed to the blistering fast <strong>Groq LLaMA 3.1 70B</strong> model for response generation. Strict backend guardrails are enforced to actively reject out-of-domain queries and ensure all AI responses are purely grounded in the provided factual context.</p>
              </section>

            </div>
          </div>
        </div>
      )}
    </div>
  )
}
