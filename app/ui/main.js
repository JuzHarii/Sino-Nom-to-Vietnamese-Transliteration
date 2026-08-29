import './style.css'

const API_BASE_URL = 'http://localhost:8000/api'

document.addEventListener('DOMContentLoaded', () => {
  const inputText = document.getElementById('input-text')
  const translateBtn = document.getElementById('translate-btn')
  const modelSelect = document.getElementById('model-select')
  
  const localOutput = document.getElementById('local-output')
  const localLoading = document.getElementById('local-loading')
  
  const hcmusOutput = document.getElementById('hcmus-output')
  const hcmusLoading = document.getElementById('hcmus-loading')

  translateBtn.addEventListener('click', async () => {
    const text = inputText.value.trim()
    if (!text) return

    const selectedModel = modelSelect.value

    // Reset outputs
    localOutput.textContent = ''
    hcmusOutput.textContent = ''
    
    // Show loading
    localLoading.classList.remove('hidden')
    hcmusLoading.classList.remove('hidden')
    translateBtn.disabled = true
    translateBtn.classList.add('opacity-75', 'cursor-not-allowed')
    translateBtn.textContent = 'Translating...'

    try {
      // Run both requests concurrently
      const [localRes, hcmusRes] = await Promise.allSettled([
        fetch(`${API_BASE_URL}/translate/local`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, model: selectedModel })
        }).then(res => res.json()),
        
        fetch(`${API_BASE_URL}/translate/hcmus`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        }).then(res => res.json())
      ])

      // Handle Local response
      if (localRes.status === 'fulfilled') {
        localOutput.textContent = localRes.value.result || 'No result'
      } else {
        localOutput.textContent = 'Error: Failed to connect to local API.'
        localOutput.classList.add('text-red-600')
      }

      // Handle HCMUS response
      if (hcmusRes.status === 'fulfilled') {
        hcmusOutput.textContent = hcmusRes.value.result || 'No result'
      } else {
        hcmusOutput.textContent = 'Error: Failed to connect to HCMUS API.'
        hcmusOutput.classList.add('text-red-600')
      }

    } catch (error) {
      console.error('Translation error:', error)
    } finally {
      localLoading.classList.add('hidden')
      hcmusLoading.classList.add('hidden')
      translateBtn.disabled = false
      translateBtn.classList.remove('opacity-75', 'cursor-not-allowed')
      translateBtn.textContent = 'Translate'
    }
  })
})
