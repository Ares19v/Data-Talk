import React, { useState, useRef, useCallback } from 'react'
import { Mic, Square, Loader2, Radio } from 'lucide-react'
import { voiceQuery } from '../api/client'

const RECORDING_LIMIT_MS = 30000 // 30 seconds

export default function VoiceRecorder({ onResult, onError, topK }) {
  const [state, setState] = useState('idle') // idle | recording | processing
  const [elapsed, setElapsed] = useState(0)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const startTimeRef = useRef(null)

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    clearInterval(timerRef.current)
  }, [])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        setState('processing')
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        try {
          const result = await voiceQuery(blob, topK)
          onResult(result)
        } catch (err) {
          onError(err?.response?.data?.detail || err.message || 'Voice query failed')
        } finally {
          setState('idle')
          setElapsed(0)
        }
      }

      recorder.start(100)
      setState('recording')
      startTimeRef.current = Date.now()

      timerRef.current = setInterval(() => {
        const secs = Math.floor((Date.now() - startTimeRef.current) / 1000)
        setElapsed(secs)
        if (Date.now() - startTimeRef.current >= RECORDING_LIMIT_MS) stopRecording()
      }, 500)
    } catch (err) {
      onError('Microphone access denied. Please allow microphone permission.')
      setState('idle')
    }
  }, [topK, onResult, onError, stopRecording])

  const handleClick = () => {
    if (state === 'idle') startRecording()
    else if (state === 'recording') stopRecording()
  }

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="relative flex items-center justify-center">

        {/* Main Button */}
        <button
          onClick={handleClick}
          disabled={state === 'processing'}
          className={`
            relative z-10 w-28 h-28 sm:w-32 sm:h-32 rounded-full flex items-center justify-center
            transition-all duration-500 ease-out transform hover:scale-105 active:scale-95
            ${state === 'recording'
              ? 'bg-gradient-to-b from-orange-500 to-red-600 shadow-[0_0_40px_rgba(249,115,22,0.6)] border border-orange-400'
              : state === 'processing'
              ? 'bg-[#3a2824] border border-[#4a3530] shadow-inner cursor-not-allowed opacity-80'
              : 'bg-gradient-to-b from-orange-400 to-orange-600 hover:from-orange-300 hover:to-orange-500 shadow-[0_10px_40px_rgba(249,115,22,0.4)] border border-orange-300/50'
            }
          `}
        >
          {state === 'processing' ? (
            <div className="flex flex-col items-center gap-2 text-orange-300">
              <Loader2 className="w-10 h-10 sm:w-12 sm:h-12 animate-spin" />
            </div>
          ) : state === 'recording' ? (
            <Square className="w-10 h-10 sm:w-12 sm:h-12 text-white fill-current" />
          ) : (
            <Mic className="w-10 h-10 sm:w-12 sm:h-12 text-white" />
          )}
        </button>
      </div>

      {/* Status Text */}
      <div className="h-6 flex items-center justify-center">
        {state === 'idle' && (
          <span className="text-sm font-medium text-orange-200/50 tracking-wide">Tap to record</span>
        )}
        {state === 'recording' && (
          <div className="flex items-center gap-2 text-orange-100 font-medium px-4 py-1.5 bg-orange-500/20 rounded-full border border-orange-500/30">
            <Radio className="w-4 h-4 animate-pulse" />
            <span>Recording {elapsed}s / 30s</span>
          </div>
        )}
        {state === 'processing' && (
          <span className="text-sm font-medium text-orange-400 tracking-wide animate-pulse">
            Transcribing & searching...
          </span>
        )}
      </div>
    </div>
  )
}
