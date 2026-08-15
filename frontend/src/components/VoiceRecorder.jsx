import React, { useState, useRef, useCallback } from 'react'
import { Mic, MicOff, Loader2, Send, Radio } from 'lucide-react'
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

      // Elapsed timer
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
    <div className="flex flex-col items-center gap-4">
      <button
        onClick={handleClick}
        disabled={state === 'processing'}
        className={`
          relative w-24 h-24 rounded-full flex items-center justify-center
          transition-all duration-300 shadow-lg
          ${state === 'recording'
            ? 'bg-red-500 hover:bg-red-600 animate-pulse'
            : state === 'processing'
            ? 'bg-slate-600 cursor-not-allowed'
            : 'bg-indigo-600 hover:bg-indigo-500'
          }
        `}
      >
        {state === 'processing' ? (
          <Loader2 className="w-10 h-10 text-white animate-spin" />
        ) : state === 'recording' ? (
          <MicOff className="w-10 h-10 text-white" />
        ) : (
          <Mic className="w-10 h-10 text-white" />
        )}

        {state === 'recording' && (
          <span className="absolute -top-1 -right-1 flex h-4 w-4">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-4 w-4 bg-red-500" />
          </span>
        )}
      </button>

      <p className="text-sm text-slate-400">
        {state === 'idle' && 'Click to speak'}
        {state === 'recording' && (
          <span className="flex items-center gap-1.5 text-red-400 font-medium">
            <Radio className="w-3.5 h-3.5" />
            Recording… {elapsed}s / 30s
          </span>
        )}
        {state === 'processing' && (
          <span className="text-indigo-400">Transcribing & searching…</span>
        )}
      </p>
    </div>
  )
}
