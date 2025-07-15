// components/MicrophoneButton.js - Working version
import React, { useRef, useEffect, useState } from 'react';
import micImg from './ai-mic-2.png'; // Make sure this image exists

const MicrophoneButton = ({ isRecording, setIsRecording, disabled, onTranscription }) => {
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const animationFrameRef = useRef(null);
  const streamRef = useRef(null);
  const [volume, setVolume] = useState(0);
  const [recordingError, setRecordingError] = useState('');

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
    if (disabled) return;

    if (isRecording) {
      console.log("Stopping recording...");
      setIsRecording(false);
      
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      
      cancelAnimationFrame(animationFrameRef.current);
      setVolume(0);
      return;
    }

    console.log("Starting recording...");
    setRecordingError('');
    setIsRecording(true);

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 44100,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      
      streamRef.current = stream;
      console.log("Got microphone access");

      // Set up MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      // Set up Web Audio API for volume visualization
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      sourceRef.current.connect(analyserRef.current);
      animateWave();

      mediaRecorder.ondataavailable = (e) => {
        console.log("Audio data available:", e.data.size);
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        console.log("MediaRecorder stopped, processing audio...");
        
        if (audioChunksRef.current.length > 0) {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          console.log("Created audio blob:", audioBlob.size, "bytes");
          
          if (onTranscription) {
            onTranscription(audioBlob);
          }
        } else {
          console.error("No audio data recorded");
          setRecordingError("No audio recorded. Please try again.");
        }
        
        // Cleanup
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        
        if (audioContextRef.current) {
          audioContextRef.current.close();
          audioContextRef.current = null;
        }
        
        cancelAnimationFrame(animationFrameRef.current);
        setVolume(0);
      };

      mediaRecorder.onerror = (e) => {
        console.error("MediaRecorder error:", e);
        setRecordingError("Recording failed. Please try again.");
        setIsRecording(false);
      };

      mediaRecorder.start();
      console.log("MediaRecorder started");
      
    } catch (err) {
      console.error("Microphone access error:", err);
      setRecordingError("Microphone access denied. Please allow microphone access and try again.");
      setIsRecording(false);
      setVolume(0);
    }
  };

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  // Wave animation styles
  const baseWaveStyle = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: '120px',
    height: '120px',
    borderRadius: '50%',
    background: 'rgba(59,130,246,0.45)',
    zIndex: 0,
    transition: 'transform 0.1s, opacity 0.1s',
  };

  // Scale wave based on volume
  const scale = 1 + Math.min(volume * 6, 1.5);
  const opacity = 0.5 + Math.min(volume * 2, 0.4);
  const waveStyle = {
    ...baseWaveStyle,
    transform: `translate(-50%, -50%) scale(${scale})`,
    opacity: opacity,
  };

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      {/* Wave animation when recording */}
      {isRecording && <div style={waveStyle}></div>}
      
      {/* Microphone button */}
      <button
        type="button"
        onClick={handleMicClick}
        disabled={disabled}
        className="rounded-full flex items-center justify-center transition-all duration-200"
        style={{ 
          width: '140px', 
          height: '140px', 
          position: 'relative', 
          zIndex: 3, 
          background: 'transparent', 
          border: 'none', 
          boxShadow: 'none', 
          padding: 0,
          opacity: disabled ? 0.5 : 1,
          cursor: disabled ? 'not-allowed' : 'pointer'
        }}
        aria-label={isRecording ? 'Stop Recording' : 'Start Recording'}
      >
        <img
          src={micImg}
          alt="Microphone"
          style={{ 
            width: 120, 
            height: 120, 
            filter: isRecording ? 'grayscale(0%)' : 'grayscale(40%)', 
            opacity: isRecording ? 1 : 0.8, 
            transition: 'filter 0.2s, opacity 0.2s' 
          }}
        />
      </button>
      
      {/* Error message */}
      {recordingError && (
        <div style={{ 
          position: 'absolute', 
          bottom: '-50px', 
          left: '50%', 
          transform: 'translateX(-50%)',
          color: '#ef4444',
          fontSize: '11px',
          textAlign: 'center',
          maxWidth: '200px'
        }}>
          {recordingError}
        </div>
      )}
    </div>
  );
};

export default MicrophoneButton;