import React, { useState, useRef } from 'react';
import { Send, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import MicrophoneButton from './MicrophoneButton';

// Enhanced language detection for better Marathi/Hindi distinction
function detectLanguage(text) {
  if (!text) return 'en';
  
  // Marathi-specific patterns and words
  const marathiPatterns = [
    /च्या/g, /त्या/g, /ह्या/g, /आहे/g, /आहेत/g,
    /केले/g, /केला/g, /केली/g, /झाले/g, /झाला/g, /झाली/g,
    /येते/g, /जाते/g, /पाहिजे/g, /लागते/g, /सांगा/g,
    /विचारा/g, /घ्या/g, /द्या/g, /करा/g, /बघा/g
  ];
  
  // Hindi-specific patterns and words
  const hindiPatterns = [
    /\bहै\b/g, /\bहैं\b/g, /\bथा\b/g, /\bथी\b/g, /\bथे\b/g,
    /करता\s+है/g, /करती\s+है/g, /करते\s+हैं/g,
    /\bचाहिए\b/g, /\bकिया\b/g, /\bगया\b/g, /\bआया\b/g
  ];
  
  // Count matches
  let marathiScore = 0;
  let hindiScore = 0;
  
  marathiPatterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) marathiScore += matches.length;
  });
  
  hindiPatterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) hindiScore += matches.length;
  });
  
  // Check for Devanagari script
  const hasDevanagari = /[\u0900-\u097F]/.test(text);
  const hasEnglish = /[A-Za-z]/.test(text);
  
  if (hasDevanagari) {
    if (marathiScore > hindiScore) {
      return 'mr';
    } else if (hindiScore > 0) {
      return 'hi';
    } else {
      // Default to Marathi if Devanagari but no clear indicators
      return 'mr';
    }
  } else if (hasEnglish) {
    return 'en';
  }
  
  return 'en'; // fallback
}

const QueryInput = ({
  onSubmit,
  isLoading,
  placeholderText = '',
  isRagBuilding,
  assistantReply,
  apiClient,
}) => {
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcriptionError, setTranscriptionError] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((inputText || '').trim() && !isLoading) {
      onSubmit((inputText || '').trim());
      setInputText('');
      setTranscriptionError(''); // Clear info after submit
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      if ((inputText || '').trim()) {
        onSubmit((inputText || '').trim());
        setInputText('');
        setTranscriptionError(''); // Clear info after submit
      }
    }
  };

  const handleTranscription = async (audioBlob) => {
    if (!audioBlob || !apiClient) {
      console.error('No audio blob or API client available');
      return;
    }

    setIsTranscribing(true);
    setTranscriptionError('');
    // Removed: setTranscriptionInfo('Processing audio...');

    try {
      console.log('Starting transcription with audio blob size:', audioBlob.size);
      
      // Use the API client's transcribe method
      const result = await apiClient.transcribeAudio(audioBlob);
      console.log('Transcription result:', result);
      
      if (result.success && result.transcription) {
        const transcription = result.transcription.trim();
        setInputText(transcription);
        
        if (textareaRef.current) {
          textareaRef.current.focus();
        }
        
      } else {
        setTranscriptionError(result.error || 'Transcription failed');
        console.error('Transcription failed:', result.error);
      }
    } catch (error) {
      setTranscriptionError('Transcription failed. Please try again.');
      console.error('Transcription error:', error);
    } finally {
      setIsTranscribing(false);
    }
  };

  // Get placeholder text based on detected input language
  const getPlaceholder = () => {
    if (inputText) {
      const lang = detectLanguage(inputText);
      if (lang === 'mr') return 'तुमचा प्रश्न येथे टाइप करा...';
      if (lang === 'hi') return 'अपना प्रश्न यहाँ टाइप करें...';
    }
    return placeholderText || 'Enter your question here... / तुमचा प्रश्न येथे टाइप करा...';
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      {/* Show loader when building RAG system */}
      {isRagBuilding && (
        <div className="flex items-center space-x-2 text-blue-600 text-lg font-semibold mb-2">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Building RAG system, please wait...</span>
        </div>
      )}

      {/* Show loader when transcribing audio */}
      {isTranscribing && (
        <div className="flex items-center space-x-2 text-blue-600 text-lg font-semibold mb-2">
          <Loader2 className="w-6 h-6 animate-spin" />
          <span>Transcribing audio...</span>
        </div>
      )}

      {/* Show transcription error */}
      {transcriptionError && (
        <div className="flex items-center space-x-2 text-red-600 text-sm font-medium mb-2 bg-red-50 px-3 py-2 rounded-lg">
          <span>⚠️ {transcriptionError}</span>
          <button 
            onClick={() => setTranscriptionError('')}
            className="text-red-400 hover:text-red-600"
          >
            ✕
          </button>
        </div>
      )}

      {/* Microphone button */}
      <div className="w-full flex justify-center mb-2">
        <MicrophoneButton
          isRecording={isRecording}
          setIsRecording={setIsRecording}
          onTranscription={handleTranscription}
          disabled={isLoading || isRagBuilding || isTranscribing}
        />
      </div>

      
      {/* Text input and send button below */}
      <form onSubmit={handleSubmit} className="w-full flex flex-col items-center">
        <textarea
          ref={textareaRef}
          className="w-full p-3 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400 mb-2"
          rows={2}
          placeholder={getPlaceholder()}
          value={inputText}
          onChange={(e) => {
            setInputText(e.target.value);
            // Removed: setTranscriptionInfo('');
          }}
          onKeyDown={handleKeyDown}
          disabled={isLoading || isRagBuilding || isTranscribing}
          style={{
            fontFamily: detectLanguage(inputText) !== 'en' 
              ? "'Noto Sans Devanagari', 'Noto Sans', sans-serif" 
              : "inherit"
          }}
        />
        <button
          type="submit"
          className="w-full py-2 px-4 rounded bg-blue-500 text-white font-semibold flex items-center justify-center disabled:opacity-50"
          disabled={isLoading || isRagBuilding || isTranscribing || !inputText.trim()}
        >
          {isLoading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : <Send className="w-5 h-5 mr-2" />}
          Send Question / प्रश्न पाठवा
        </button>
      </form>

      {/* Only render assistant reply as markdown if present */}
      {assistantReply && (
        <div className="w-full max-w-xl mt-4 prose lg:prose-xl">
          <ReactMarkdown>{assistantReply}</ReactMarkdown>
        </div>
      )}
    </div>
  );
};

export default QueryInput;