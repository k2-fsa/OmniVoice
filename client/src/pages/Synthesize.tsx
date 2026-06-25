import React, { useState } from 'react';
import axios from 'axios';
import { Button } from '@/components/ui/button';
import { Upload } from 'lucide-react';
import { toast } from 'sonner';

export default function Synthesize() {
  const [refFile, setRefFile] = useState<File | null>(null);
  const [texts, setTexts] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleFile = (f: File | null) => setRefFile(f);

  const submit = async () => {
    if (!texts.trim()) return toast.error('Provide one or more sentences');
    setIsLoading(true);
    try {
      const form = new FormData();
      form.append('texts', texts);
      if (refFile) form.append('ref_audio', refFile);
      const resp = await axios.post('/api/synthesize', form, {
        responseType: 'blob',
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      // Download the zip
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
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">Voice Cloning Synthesize</h1>
      <div className="mb-4">
        <label className="block mb-2">Reference audio (optional)</label>
        <input type="file" accept="audio/*" onChange={(e) => handleFile(e.target.files?.[0] ?? null)} />
      </div>
      <div className="mb-4">
        <label className="block mb-2">Sentences (one per line)</label>
        <textarea rows={6} className="w-full p-2 border rounded" value={texts} onChange={(e) => setTexts(e.target.value)} />
      </div>
      <div className="flex gap-2">
        <Button onClick={submit} disabled={isLoading}>
          {isLoading ? 'Synthesizing...' : 'Synthesize'}
        </Button>
      </div>
    </div>
  );
}
