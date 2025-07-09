import React, { useRef, useEffect, useState } from 'react';
import {
  Volume2, VolumeX, Copy, CheckCircle, Loader2,
  Pause, RotateCcw, Play, Download
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './AnswerSection.css';

function detectLang(text) {
  if (/[\u0900-\u097F]/.test(text)) {
    if (text.includes('च्या') || text.includes('आहे')) return 'mr';
    return 'hi';
  }
  if (/[A-Za-z]/.test(text)) return 'en';
  return 'en';
}

const AnswerSection = ({ answer, question, onGenerateTTS, audioUrl, autoPlay, onAudioUrl }) => {
  const [isPlayingTTS, setIsPlayingTTS] = useState(false);
  const [isGeneratingTTS, setIsGeneratingTTS] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [audioProgress, setAudioProgress] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [hasAutoplayed, setHasAutoplayed] = useState(false);

  const audioRef = useRef(null);
  const shouldAutoplayRef = useRef(false);

  const cleanAnswer = (text) =>
    typeof text === 'string' ? text.replace(/^\[Cached\]\s*/, '') : JSON.stringify(text);

  const handleCopyAnswer = async () => {
    try {
      await navigator.clipboard.writeText(cleanAnswer(answer));
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (error) {
      console.error('Copy failed:', error);
    }
  };

  useEffect(() => {
    if (!audioRef.current) {
      audioRef.current = new Audio();
    }

    const audio = audioRef.current;

    const handlePlay = () => setIsPlayingTTS(true);
    const handlePause = () => setIsPlayingTTS(false);
    const handleEnded = () => setIsPlayingTTS(false);
    const handleTimeUpdate = () => {
      setAudioProgress(audio.currentTime);
      setAudioDuration(audio.duration || 0);
    };

    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      audio.pause();
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !audioUrl) return;

    const tryAutoplay = async () => {
      if ((autoPlay || shouldAutoplayRef.current) && !hasAutoplayed) {
        try {
          await audio.play();
          setHasAutoplayed(true);
          shouldAutoplayRef.current = false;
        } catch (err) {
          console.warn('Autoplay blocked:', err);
          shouldAutoplayRef.current = false;
        }
      }
    };

    const setupAudio = () => {
      if (audio.src !== audioUrl) {
        audio.src = audioUrl;
        audio.load();
        setAudioProgress(0);
        setAudioDuration(0);
      }

      audio.playbackRate = playbackRate;

      if (audio.readyState >= 3) {
        tryAutoplay();
      } else {
        audio.addEventListener('canplaythrough', tryAutoplay, { once: true });
      }
    };

    setupAudio();

    return () => {
      audio.removeEventListener('canplaythrough', tryAutoplay);
    };
  }, [audioUrl, autoPlay, playbackRate, hasAutoplayed]);

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }

    setAudioProgress(0);
    setAudioDuration(0);
    setIsPlayingTTS(false);
    setHasAutoplayed(false);
    shouldAutoplayRef.current = true;
  }, [answer, audioUrl]);

  const handlePlayPause = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (audio.paused) {
      if (audio.readyState < 2) {
        await new Promise(resolve =>
          audio.addEventListener('canplaythrough', resolve, { once: true })
        );
      }
      try {
        await audio.play();
      } catch (err) {
        console.warn('Play error:', err);
      }
    } else {
      audio.pause();
    }
  };

  const handleReplay = async () => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = 0;
    try {
      await audio.play();
    } catch (err) {
      console.warn('Replay error:', err);
    }
  };

  const handleSeek = (e) => {
    const audio = audioRef.current;
    if (!audio || !audioDuration) return;
    const rect = e.target.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    const seekTime = percent * audioDuration;
    audio.currentTime = seekTime;
    setAudioProgress(seekTime);
  };

  // Language check for TTS (strict)
  const lang = detectLang(cleanAnswer(answer));
  const isSupportedLang = lang === 'en' || lang === 'hi' || lang === 'mr';

  const handleGenerateTTS = async () => {
    if (!isSupportedLang) {
      alert('❌ Only English, Hindi, and Marathi are supported for TTS.');
      return;
    }
    setIsGeneratingTTS(true);
    try {
      const cleanedAnswer = cleanAnswer(answer);
      const ttsResult = await onGenerateTTS(cleanedAnswer);
      if (ttsResult?.audio_base64) {
        const audioBlob = new Blob(
          [Uint8Array.from(atob(ttsResult.audio_base64), c => c.charCodeAt(0))],
          { type: 'audio/wav' }
        );
        const url = URL.createObjectURL(audioBlob);
        if (typeof onAudioUrl === 'function') {
          onAudioUrl(url);
        }

        const audio = audioRef.current;
        audio.src = url;
        audio.load();
        audio.playbackRate = playbackRate;
        setAudioProgress(0);
        setAudioDuration(0);
        shouldAutoplayRef.current = true;
      }
    } catch (error) {
      console.error('TTS generation failed:', error);
      alert('❌ TTS generation failed. Please use English, Hindi, or Marathi.');
    } finally {
      setIsGeneratingTTS(false);
    }
  };

  if (!answer) return null;

  return (
    <div className="flex flex-col items-end space-y-2 w-full">
      {question && (
        <div className="flex w-full justify-end">
          <div className="bg-blue-50 text-blue-900 rounded-2xl px-5 py-3 shadow max-w-2xl text-right">
            <span className="block font-semibold text-blue-700 mb-1">You</span>
            <span className="whitespace-pre-wrap break-words">{question}</span>
          </div>
        </div>
      )}
      <div className="flex w-full justify-start">
        <div className="bg-green-50 text-green-900 rounded-2xl px-5 py-3 shadow max-w-2xl relative">
          <span className="block font-semibold text-green-700 mb-1">AI Assistant</span>
          <div className="prose prose-blue max-w-none">
            <div className="text-gray-700 leading-relaxed">
              <div className="markdown-content" lang={lang}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    h1: (props) => <h1 className="text-2xl font-bold mb-4 mt-8" {...props} />,
                    h2: (props) => <h2 className="text-xl font-bold mb-3 mt-6" {...props} />,
                    h3: (props) => <h3 className="text-lg font-bold mb-2 mt-5" {...props} />,
                    p: (props) => <p className="mb-4" {...props} />,
                    ul: (props) => <ul className="list-disc list-inside mb-6 space-y-2" {...props} />,
                    ol: (props) => <ol className="list-decimal list-inside mb-6 space-y-2" {...props} />,
                    li: (props) => <li className="mb-2" {...props} />,
                    strong: (props) => <strong className="font-bold" {...props} />,
                    em: (props) => <em className="italic" {...props} />,
                    blockquote: (props) => <blockquote className="border-l-4 border-blue-500 pl-4 italic my-6" {...props} />,
                  }}
                >
                  {cleanAnswer(answer)}
                </ReactMarkdown>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex flex-col space-y-1 mt-2">
            {!isSupportedLang && (
              <div className="text-red-600 font-semibold text-sm mb-2">
                ❌ Only English, Hindi, and Marathi are supported for audio. TTS is disabled for this answer.
              </div>
            )}
            <div className="flex items-center space-x-2">
              <button
                onClick={async () => {
                  if (!isSupportedLang) return;
                  if (audioUrl) {
                    await handlePlayPause();
                  } else {
                    await handleGenerateTTS();
                  }
                }}
                disabled={isGeneratingTTS || !isSupportedLang}
                className="p-2 rounded-lg bg-green-100 text-green-600 hover:bg-green-200 transition-colors disabled:opacity-50"
                title={isSupportedLang ? (isPlayingTTS ? 'Pause Audio' : 'Play Audio') : 'TTS not allowed for this language'}
              >
                {isGeneratingTTS ? <Loader2 className="w-4 h-4 animate-spin" /> :
                  isPlayingTTS ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>

              <button
                onClick={isSupportedLang ? handleReplay : undefined}
                disabled={!audioUrl || !isSupportedLang}
                className="p-2 rounded-lg bg-purple-100 text-purple-600 hover:bg-purple-200 transition-colors disabled:opacity-50"
                title={isSupportedLang ? 'Replay Audio' : 'TTS not allowed for this language'}
              >
                <RotateCcw className="w-4 h-4" />
              </button>

              <button
                onClick={handleCopyAnswer}
                className="p-2 rounded-lg bg-blue-100 text-blue-600 hover:bg-blue-200 transition-colors"
                title="Copy Answer"
              >
                {isCopied ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              </button>

              <select
                value={playbackRate}
                onChange={(e) => setPlaybackRate(Number(e.target.value))}
                className="p-1 rounded bg-gray-100 text-gray-700 text-xs border border-gray-200"
                title="Playback Speed"
                disabled={!isSupportedLang}
              >
                <option value={0.75}>0.75x</option>
                <option value={1.0}>1x</option>
                <option value={1.25}>1.25x</option>
                <option value={1.5}>1.5x</option>
                <option value={2.0}>2x</option>
              </select>
            </div>

            {/* Audio Progress Bar */}
            {audioUrl && isSupportedLang && (
              <>
                <div className="flex items-center space-x-2 w-full">
                  <span className="text-xs text-gray-500 w-10 text-right">
                    {new Date(audioProgress * 1000).toISOString().substr(14, 5)}
                  </span>
                  <div
                    className="flex-1 h-2 bg-gray-200 rounded cursor-pointer relative"
                    onClick={handleSeek}
                  >
                    <div
                      className="h-2 bg-green-400 rounded"
                      style={{ width: `${(audioProgress / (audioDuration || 1)) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-500 w-10 text-left">
                    {audioDuration ? new Date(audioDuration * 1000).toISOString().substr(14, 5) : '00:00'}
                  </span>
                </div>
                <div className="flex justify-end mt-1">
                  <a
                    href={audioUrl}
                    download="answer-audio.wav"
                    className="inline-flex items-center px-3 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 text-xs font-medium transition-colors"
                    title="Download Audio"
                  >
                    <Download className="w-4 h-4" />
                  </a>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnswerSection;
