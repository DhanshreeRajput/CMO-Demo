import React, { useState, useRef } from 'react';
import { Send, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import MicrophoneButton from './MicrophoneButton';

const QueryInput = ({
  onSubmit,
  isLoading,
  placeholderText = '',
  isRagBuilding,
  assistantReply,
}) => {
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const textareaRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const socketRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if ((inputText || '').trim() && !isLoading) {
      onSubmit((inputText || '').trim());
      setInputText('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      if ((inputText || '').trim()) {
        onSubmit((inputText || '').trim());
        setInputText('');
      }
    }
  };

  const handleTranscription = async (audioBlob) => {
    if (!audioBlob) return;
    setIsTranscribing(true);
    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'audio.wav');
      const response = await fetch('http://localhost:8000/transcribe/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (response.ok && data.transcription) {
        setInputText(data.transcription);
        if (textareaRef.current) textareaRef.current.focus();
      } else {
        setInputText('');
        alert(data.error || 'Transcription failed.');
      }
    } catch (err) {
      setInputText('');
      alert('Transcription failed.');
    } finally {
      setIsTranscribing(false);
    }
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
          placeholder={placeholderText || 'Enter your question here...'}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading || isRagBuilding || isTranscribing}
        />
        <button
          type="submit"
          className="w-full py-2 px-4 rounded bg-blue-500 text-white font-semibold flex items-center justify-center disabled:opacity-50"
          disabled={isLoading || isRagBuilding || isTranscribing || !inputText.trim()}
        >
          {isLoading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : <Send className="w-5 h-5 mr-2" />}
          Send Question
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