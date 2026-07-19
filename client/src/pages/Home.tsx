import axios from 'axios';
import { Button } from "@/components/ui/button";
import React, { useRef, useState } from 'react';
import { toast } from 'sonner';

const Upload = (props: React.SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);

const Music = (props: React.SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    <path d="M9 18V5l12-2v13" />
    <circle cx="6" cy="18" r="3" />
    <circle cx="18" cy="16" r="3" />
  </svg>
);

const FileText = (props: React.SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
    <line x1="10" y1="9" x2="8" y2="9" />
  </svg>
);

/**
 * Hebrew Transcription App - File Upload Interface
 * Replaces URL parameter-based loading with proper file upload using FileReader API
 * Supports drag-and-drop for both audio and transcription files
 */
export default function Home() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [transcriptionText, setTranscriptionText] = useState<string | null>(null);
  const [sentenceList, setSentenceList] = useState('');
  const [language, setLanguage] = useState<'en' | 'he'>('en');
  const [isLoading, setIsLoading] = useState(false);
  
  const audioInputRef = useRef<HTMLInputElement>(null);
  const transcriptionInputRef = useRef<HTMLInputElement>(null);
  const audioDropZoneRef = useRef<HTMLDivElement>(null);
  const transcriptionDropZoneRef = useRef<HTMLDivElement>(null);

  const AUDIO_TYPES = ['.wav', '.mp3', '.ogg', '.flac'];
  const TRANSCRIPTION_TYPES = ['.txt'];

  /**
   * Handle audio file selection
   */
  const handleAudioFileSelect = async (file: File) => {
    // Validate file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!AUDIO_TYPES.includes(fileExtension)) {
      toast.error(`Invalid audio format. Supported: ${AUDIO_TYPES.join(', ')}`);
      return;
    }

    // Validate file size (max 100MB)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      toast.error('Audio file is too large. Maximum size: 100MB');
      return;
    }

    setIsLoading(true);
    try {
      setAudioFile(file);
      // Create object URL for audio playback
      const url = URL.createObjectURL(file);
      setAudioUrl(url);
      toast.success(`Audio file loaded: ${file.name}`);
    } catch (error) {
      console.error('Error loading audio file:', error);
      toast.error('Failed to load audio file');
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Handle transcription file selection
   */
  const handleTranscriptionFileSelect = async (file: File) => {
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.txt')) {
      toast.error('Invalid transcription format. Please upload a .txt file');
      return;
    }

    setIsLoading(true);
    try {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        setTranscriptionText(text);
        toast.success(`Transcription file loaded: ${file.name}`);
      };
      reader.onerror = () => {
        toast.error('Failed to read transcription file');
      };
      reader.readAsText(file);
    } catch (error) {
      console.error('Error loading transcription file:', error);
      toast.error('Failed to load transcription file');
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Handle file input change
   */
  const handleAudioInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleAudioFileSelect(file);
    }
  };

  const handleTranscriptionInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleTranscriptionFileSelect(file);
    }
  };

  /**
   * Handle drag and drop for audio
   */
  const handleAudioDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    audioDropZoneRef.current?.classList.add('border-blue-500', 'bg-blue-50', 'dark:bg-blue-950');
  };

  const handleAudioDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    audioDropZoneRef.current?.classList.remove('border-blue-500', 'bg-blue-50', 'dark:bg-blue-950');
  };

  const handleAudioDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    audioDropZoneRef.current?.classList.remove('border-blue-500', 'bg-blue-50', 'dark:bg-blue-950');
    
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleAudioFileSelect(file);
    }
  };

  /**
   * Handle drag and drop for transcription
   */
  const handleTranscriptionDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    transcriptionDropZoneRef.current?.classList.add('border-green-500', 'bg-green-50', 'dark:bg-green-950');
  };

  const handleTranscriptionDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    transcriptionDropZoneRef.current?.classList.remove('border-green-500', 'bg-green-50', 'dark:bg-green-950');
  };

  const handleTranscriptionDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    transcriptionDropZoneRef.current?.classList.remove('border-green-500', 'bg-green-50', 'dark:bg-green-950');
    
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleTranscriptionFileSelect(file);
    }
  };

  /**
   * Clear audio file
   */
  const clearAudio = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioFile(null);
    setAudioUrl(null);
    if (audioInputRef.current) {
      audioInputRef.current.value = '';
    }
    toast.info('Audio file cleared');
  };

  /**
   * Clear transcription file
   */
  const clearTranscription = () => {
    setTranscriptionText(null);
    if (transcriptionInputRef.current) {
      transcriptionInputRef.current.value = '';
    }
    toast.info('Transcription file cleared');
  };

  const handleSynthesize = async () => {
    if (!sentenceList.trim()) {
      toast.error('Enter one or more sentences to synthesize');
      return;
    }

    setIsLoading(true);
    try {
      const form = new FormData();
      form.append('texts', sentenceList.trim());
      form.append('language', language);
      if (audioFile) {
        form.append('ref_audio', audioFile);
      }
      if (transcriptionText) {
        form.append('ref_text', transcriptionText);
      }

      const resp = await axios.post('/api/synthesize', form, {
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([resp.data]));
      const a = document.createElement('a');
      a.href = url;
      a.download = 'synth.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      toast.success('Synthesis complete — downloaded zip');
    } catch (err: any) {
      console.error(err);
      toast.error(err?.response?.data?.error || err.message || 'Synthesis failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <main className="flex-1 p-6 md:p-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-2 text-slate-900 dark:text-slate-50">
              Hebrew Transcription App
            </h1>
            <p className="text-slate-600 dark:text-slate-400">
              Upload your audio file and transcription to get started
            </p>
          </div>

          {/* Upload Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {/* Audio Upload */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Music className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
                  Audio File
                </h2>
              </div>

              {/* Audio Drop Zone */}
              <div
                ref={audioDropZoneRef}
                onDragOver={handleAudioDragOver}
                onDragLeave={handleAudioDragLeave}
                onDrop={handleAudioDrop}
                className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-lg p-8 text-center cursor-pointer transition-all hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-950"
                onClick={() => audioInputRef.current?.click()}
              >
                <Upload className="w-12 h-12 text-slate-400 dark:text-slate-500 mx-auto mb-3" />
                <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Drag and drop your audio file here
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
                  or click to browse
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Supported: WAV, MP3, OGG, FLAC (max 100MB)
                </p>
              </div>

              {/* Audio File Input */}
              <input
                ref={audioInputRef}
                type="file"
                accept=".wav,.mp3,.ogg,.flac,audio/wav,audio/mpeg,audio/ogg,audio/flac"
                onChange={handleAudioInputChange}
                className="hidden"
              />

              {/* Audio Status */}
              {audioFile && (
                <div className="bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                  <p className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-2">
                    ✓ Audio loaded: {audioFile.name}
                  </p>
                  <p className="text-xs text-blue-700 dark:text-blue-300 mb-3">
                    Size: {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={clearAudio}
                    className="w-full"
                  >
                    Clear Audio
                  </Button>
                </div>
              )}
            </div>

            {/* Transcription Upload */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-green-600 dark:text-green-400" />
                <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-50">
                  Transcription File
                </h2>
              </div>

              {/* Transcription Drop Zone */}
              <div
                ref={transcriptionDropZoneRef}
                onDragOver={handleTranscriptionDragOver}
                onDragLeave={handleTranscriptionDragLeave}
                onDrop={handleTranscriptionDrop}
                className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-lg p-8 text-center cursor-pointer transition-all hover:border-green-500 hover:bg-green-50 dark:hover:bg-green-950"
                onClick={() => transcriptionInputRef.current?.click()}
              >
                <Upload className="w-12 h-12 text-slate-400 dark:text-slate-500 mx-auto mb-3" />
                <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Drag and drop your transcription file here
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
                  or click to browse
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Supported: TXT files
                </p>
              </div>

              {/* Transcription File Input */}
              <input
                ref={transcriptionInputRef}
                type="file"
                accept=".txt,text/plain"
                onChange={handleTranscriptionInputChange}
                className="hidden"
              />

              {/* Transcription Status */}
              {transcriptionText && (
                <div className="bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded-lg p-4">
                  <p className="text-sm font-medium text-green-900 dark:text-green-100 mb-3">
                    ✓ Transcription loaded
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={clearTranscription}
                    className="w-full"
                  >
                    Clear Transcription
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* Content Display Section */}
          {(audioUrl || transcriptionText) && (
            <div className="space-y-6">
              {/* Audio Playback */}
              {audioUrl && (
                <div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                  <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-slate-50">
                    Audio Playback
                  </h2>
                  <audio
                    controls
                    src={audioUrl}
                    className="w-full rounded-lg"
                  />
                </div>
              )}

              {/* Transcription Display */}
              {transcriptionText && (
                <div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 p-6">
                  <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-slate-50">
                    Transcription
                  </h2>
                  <div className="p-4 border border-slate-200 dark:border-slate-600 rounded-lg bg-slate-50 dark:bg-slate-900 whitespace-pre-wrap text-slate-700 dark:text-slate-300 font-mono text-sm max-h-96 overflow-y-auto">
                    {transcriptionText}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 p-6">
            <h2 className="text-2xl font-semibold mb-4 text-slate-900 dark:text-slate-50">
              Synthesize Sentences
            </h2>
            <div className="grid gap-4 sm:grid-cols-2 mb-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Language
                </label>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value as 'en' | 'he')}
                  className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-900 shadow-sm outline-none transition hover:border-slate-400 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                >
                  <option value="en">English</option>
                  <option value="he">Hebrew</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Sentence list
                </label>
                <textarea
                  rows={6}
                  value={sentenceList}
                  onChange={(e) => setSentenceList(e.target.value)}
                  placeholder="One sentence per line"
                  className="min-h-[170px] w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-slate-900 shadow-sm outline-none transition hover:border-slate-400 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
                />
              </div>
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
              Your transcription will be used as reference text if a transcription file is loaded. The selected reference audio and language are used to generate the sentences listed above.
            </p>
            <Button onClick={handleSynthesize} disabled={isLoading} className="w-full md:w-auto">
              {isLoading ? 'Synthesizing...' : 'Generate voice for sentences'}
            </Button>
          </div>

          {/* Empty State */}
          {!audioUrl && !transcriptionText && (
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 p-12 text-center">
              <Music className="w-16 h-16 text-slate-300 dark:text-slate-600 mx-auto mb-4" />
              <p className="text-lg font-medium text-slate-600 dark:text-slate-400 mb-2">
                No files loaded yet
              </p>
              <p className="text-sm text-slate-500 dark:text-slate-500">
                Upload an audio file and transcription to get started. Use drag-and-drop or click to browse.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
