import React, { useRef, useEffect, useState } from 'react';
import micImg from './ai-mic-2.png'; // Place ai-mic.png in src/components/

const ALLOWED_LANGS = ['en-US', 'hi-IN', 'mr-IN'];
const DEFAULT_LANG = 'en-US';

const baseWaveStyle = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: '120px',
  height: '120px',
  borderRadius: '50%',
  background: 'rgba(59,130,246,0.45)', // Increased alpha for darker wave
  zIndex: 0,
  transition: 'transform 0.1s, opacity 0.1s',
};

const keyframes = `
@keyframes mic-wave {
  0% { transform: translate(-50%, -50%) scale(1); opacity: 0.7; }
  70% { transform: translate(-50%, -50%) scale(1.5); opacity: 0.2; }
  100% { transform: translate(-50%, -50%) scale(2); opacity: 0; }
}`;

const MicrophoneButton = ({ isRecording, setIsRecording, disabled, onTranscription }) => {
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const animationFrameRef = useRef(null);
  const [volume, setVolume] = useState(0);

  useEffect(() => {
    // Inject keyframes for fallback animation (not used for volume now)
    if (!document.getElementById('mic-wave-keyframes')) {
      const style = document.createElement('style');
      style.id = 'mic-wave-keyframes';
      style.innerHTML = keyframes;
      document.head.appendChild(style);
    }
  }, []);

  const animateWave = () => {
    if (!analyserRef.current) return;
    const dataArray = new Uint8Array(analyserRef.current.fftSize);
    analyserRef.current.getByteTimeDomainData(dataArray);
    // Calculate RMS (root mean square) for volume
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const val = (dataArray[i] - 128) / 128;
      sum += val * val;
    }
    const rms = Math.sqrt(sum / dataArray.length);
    setVolume(rms);
    animationFrameRef.current = requestAnimationFrame(animateWave);
  };

  const handleMicClick = async () => {
    if (isRecording) {
      setIsRecording(false);
      if (mediaRecorderRef.current) mediaRecorderRef.current.stop();
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      cancelAnimationFrame(animationFrameRef.current);
      setVolume(0);
      return;
    }
    setIsRecording(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new window.MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      // Web Audio API for volume
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      sourceRef.current.connect(analyserRef.current);
      animateWave();

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        if (onTranscription) onTranscription(audioBlob);
        stream.getTracks().forEach(track => track.stop());
        if (audioContextRef.current) {
          audioContextRef.current.close();
          audioContextRef.current = null;
        }
        cancelAnimationFrame(animationFrameRef.current);
        setVolume(0);
      };

      mediaRecorder.start();
    } catch (err) {
      alert('Microphone access denied or not available.');
      setIsRecording(false);
      setVolume(0);
    }
  };

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  // Scale wave based on volume (min 1, max 2.5)
  const scale = 1 + Math.min(volume * 6, 1.5);
  const opacity = 0.5 + Math.min(volume * 2, 0.4); // Increased base opacity for darker effect
  const waveStyle = {
    ...baseWaveStyle,
    transform: `translate(-50%, -50%) scale(${scale})`,
    opacity: opacity,
  };

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      {isRecording && <div style={waveStyle}></div>}
      <button
        type="button"
        onClick={handleMicClick}
        disabled={disabled}
        className={`rounded-full flex items-center justify-center transition-all duration-200`}
        style={{ width: '140px', height: '140px', position: 'relative', zIndex: 3, background: 'transparent', border: 'none', boxShadow: 'none', padding: 0 }}
        aria-label={isRecording ? 'Stop Recording' : 'Start Recording'}
      >
        {/* Use AI mic image instead of emoji, XXL size */}
        <img
          src={micImg}
          alt="Mic"
          style={{ width: 120, height: 120, filter: isRecording ? 'grayscale(0%)' : 'grayscale(40%)', opacity: isRecording ? 1 : 0.8, transition: 'filter 0.2s, opacity 0.2s' }}
        />
      </button>
    </div>
  );
};

export default MicrophoneButton;

