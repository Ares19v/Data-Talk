import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  timeout: 60000,
})

/**
 * Send a text query to the RAG backend
 * @param {string} query
 * @param {number} topK
 * @returns {Promise<object>} RAGResponse
 */
export async function textQuery(query, topK = 5) {
  const { data } = await api.post('/query', {
    query,
    top_k: topK,
    include_sources: true,
  })
  return data
}

/**
 * Send an audio blob to the RAG backend (voice pipeline)
 * @param {Blob} audioBlob
 * @param {number} topK
 * @returns {Promise<object>} RAGResponse
 */
export async function voiceQuery(audioBlob, topK = 5) {
  const form = new FormData()
  form.append('audio', audioBlob, 'recording.wav')
  form.append('top_k', String(topK))
  form.append('include_sources', 'true')

  const { data } = await api.post('/voice-query', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

/**
 * Health check
 */
export async function healthCheck() {
  const { data } = await api.get('/health')
  return data
}
