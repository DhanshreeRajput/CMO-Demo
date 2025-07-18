// services/ApiClient.js - EXACT same pattern as Streamlit api_transcribe_audio and api_generate_audio_response

class ApiClient {
  constructor(baseURL = '') {
    this.baseURL = baseURL;
    this.sessionId = this.generateSessionId();
  }

  generateSessionId() {
    return 'react_' + Math.random().toString(36).substr(2, 9);
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.baseURL}/health/`);
      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  async uploadFiles(pdfFile, txtFile) {
    const formData = new FormData();
    
    if (pdfFile) {
      formData.append('pdf_file', pdfFile);
    }
    if (txtFile) {
      formData.append('txt_file', txtFile);
    }
    
    formData.append('session_id', this.sessionId);

    try {
      const response = await fetch(`${this.baseURL}/upload/`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      return data;
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  }

  async query(inputText, modelKey) {
    try {
      const response = await fetch(`${this.baseURL}/query/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input_text: inputText,
          model: 'llama-3.3-70b-versatile',
          enhanced_mode: true,
          voice_lang_pref: 'auto',
          session_id: this.sessionId,
          model_key: modelKey
        }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        if (response.status === 429) {
          throw new Error(
            data.reply ||
            'Unable to answer right now, please try again after sometime. For more details, please contact the 104/102 helpline numbers.'
          );
        }
        throw new Error(data.reply || data.error || 'Query failed');
      }

      return data;
    } catch (error) {
      console.error('Query error:', error);
      throw error;
    }
  }

  // EXACT same as Streamlit api_transcribe_audio function
  async transcribeAudio(audioBlob) {
    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'recording.wav');

      const response = await fetch(`${this.baseURL}/transcribe/`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        // Return in same format as Streamlit (success, result) tuple
        return {
          success: false,
          error: data.error || 'Transcription failed'
        };
      }

      // Return in same format as Streamlit (success, result) tuple
      return {
        success: data.success,
        transcription: data.transcription
      };
    } catch (error) {
      console.error('Transcription error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  // EXACT same as Streamlit api_generate_audio_response function
  async generateTTS(text, langPreference = "auto") {
    try {
      const formData = new FormData();
      formData.append('text', text);
      formData.append('lang_preference', langPreference);

      const response = await fetch(`${this.baseURL}/tts/`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        return {
          success: false,
          error: data.error || 'TTS generation failed'
        };
      }

      // Return in same format as Streamlit (audio_bytes, lang_used, cache_hit) but as object
      return {
        success: data.success,
        audio_base64: data.audio_base64,
        lang_used: data.lang_used,
        cache_hit: data.cache_hit
      };
    } catch (error) {
      console.error('TTS error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getChatHistory() {
    try {
      const response = await fetch(`${this.baseURL}/chat-history/?session_id=${this.sessionId}`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error('Failed to fetch chat history');
      }

      return data.chat_history || [];
    } catch (error) {
      console.error('Chat history error:', error);
      return [];
    }
  }

  async clearChatHistory() {
    try {
      const response = await fetch(`${this.baseURL}/sessions/${this.sessionId}`, {
        method: 'DELETE',
      });
      
      return response.ok;
    } catch (error) {
      console.error('Clear chat history error:', error);
      return false;
    }
  }
}

export default ApiClient;