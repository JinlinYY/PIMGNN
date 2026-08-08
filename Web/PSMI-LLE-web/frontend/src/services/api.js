import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE || '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
})

api.interceptors.request.use(
  config => config,
  error => Promise.reject(error)
)

api.interceptors.response.use(
  response => response,
  error => {
    if (error.response) {
      console.error('API Error:', error.response.data)
    } else if (error.code === 'ECONNABORTED') {
      console.error('Request timeout')
    } else {
      console.error('Network error:', error.message)
    }
    return Promise.reject(error)
  }
)

export const predictLLE = async (data) => api.post('/predict', data)
export const healthCheck = async () => api.get('/health')
export const validateSmiles = async (smiles) => api.post('/validate-smiles', { smiles })
export const loadExplainability = async () => api.get('/explain/summary')

export default {
  predict: predictLLE,
  health: healthCheck,
  validateSmiles,
  loadExplainability,
}
