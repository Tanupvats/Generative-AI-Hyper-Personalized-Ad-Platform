import React, { useState } from 'react';
import { Upload, Video, User, Phone, Briefcase, Mail, MessageSquare, Loader2, PlayCircle } from 'lucide-react';

export default function App() {
  const [formData, setFormData] = useState({
    name: '',
    phone: '',
    designation: '',
    email: '',
    dialog: ''
  });
  const [videoFile, setVideoFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [outputVideoUrl, setOutputVideoUrl] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setVideoFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!videoFile) {
      alert("Please upload a source video.");
      return;
    }

    setIsProcessing(true);
    setOutputVideoUrl(null);

    // Mocking the pipeline steps for the UI visualization
    const steps = [
      "Generating base speech with gTTS...",
      "Applying RVC Voice Style Transfer...",
      "Syncing lips with Wav2Lip...",
      "Finalizing video..."
    ];

    for (let i = 0; i < steps.length; i++) {
      setStatusMessage(steps[i]);
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time
    }

    // In a real app, this would be a FormData POST request to our FastAPI backend
    /*
      const submitData = new FormData();
      Object.keys(formData).forEach(key => submitData.append(key, formData[key]));
      submitData.append('video', videoFile);
      const response = await fetch('/api/generate-ad', { method: 'POST', body: submitData });
      const blob = await response.blob();
      setOutputVideoUrl(URL.createObjectURL(blob));
    */

    setStatusMessage("Video generated successfully!");
    // Mock output video (placeholder)
    setOutputVideoUrl("https://www.w3schools.com/html/mov_bbb.mp4"); 
    setIsProcessing(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Header */}
      <header className="bg-indigo-600 text-white py-6 shadow-md">
        <div className="max-w-6xl mx-auto px-6">
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <Video className="w-8 h-8" />
            Hyper-Personalized Ad Portal
          </h1>
          <p className="mt-2 text-indigo-100">Generative AI-driven Voice & Lip Sync Pipeline (gTTS + RVC + Wav2Lip)</p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-2 gap-10">
        
        {/* Input Form */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8">
          <h2 className="text-xl font-semibold mb-6 border-b pb-2">Client Configuration</h2>
          <form onSubmit={handleSubmit} className="space-y-5">
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2"><User className="w-4 h-4"/> Full Name</label>
                <input required type="text" name="name" value={formData.name} onChange={handleInputChange} className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none" placeholder="John Doe" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2"><Phone className="w-4 h-4"/> Phone Number</label>
                <input required type="tel" name="phone" value={formData.phone} onChange={handleInputChange} className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none" placeholder="+1 234 567 8900" />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2"><Briefcase className="w-4 h-4"/> Designation</label>
                <input required type="text" name="designation" value={formData.designation} onChange={handleInputChange} className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none" placeholder="Regional Manager" />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2"><Mail className="w-4 h-4"/> Email Address</label>
                <input required type="email" name="email" value={formData.email} onChange={handleInputChange} className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none" placeholder="john@company.com" />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2"><MessageSquare className="w-4 h-4"/> Custom Dialog</label>
              <textarea required name="dialog" value={formData.dialog} onChange={handleInputChange} rows="3" className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none resize-none" placeholder="Welcome to our summer sale! Get 20% off using code SUMMER20."></textarea>
              <p className="text-xs text-slate-500 mt-1">Note: Name and designation will be automatically prepended to the dialog.</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1 flex items-center gap-2"><Upload className="w-4 h-4"/> Source Video (Actor)</label>
              <input required type="file" accept="video/mp4" onChange={handleFileChange} className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100" />
            </div>

            <button disabled={isProcessing} type="submit" className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 rounded-lg transition-colors flex justify-center items-center gap-2 disabled:bg-indigo-400">
              {isProcessing ? <Loader2 className="w-5 h-5 animate-spin" /> : <PlayCircle className="w-5 h-5" />}
              {isProcessing ? 'Processing Pipeline...' : 'Generate Personalized Ad'}
            </button>
          </form>
        </div>

        {/* Output Section */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 flex flex-col items-center justify-center min-h-[500px]">
          {isProcessing ? (
            <div className="text-center">
              <div className="relative w-24 h-24 mx-auto mb-6">
                <div className="absolute inset-0 border-4 border-indigo-100 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin"></div>
              </div>
              <h3 className="text-lg font-medium text-slate-800 animate-pulse">{statusMessage}</h3>
              <p className="text-sm text-slate-500 mt-2">GPU instances are spinning up...</p>
            </div>
          ) : outputVideoUrl ? (
            <div className="w-full">
              <h2 className="text-xl font-semibold mb-4 text-green-600 flex items-center gap-2">
                <span className="w-3 h-3 bg-green-500 rounded-full inline-block"></span>
                Generation Complete
              </h2>
              <video src={outputVideoUrl} controls className="w-full rounded-lg shadow-md border bg-black aspect-video"></video>
              <div className="mt-4 flex gap-4">
                <button className="flex-1 bg-slate-100 hover:bg-slate-200 text-slate-800 font-medium py-2 rounded-lg transition-colors">Download .mp4</button>
                <button className="flex-1 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-medium py-2 rounded-lg transition-colors">Send Email to Lead</button>
              </div>
            </div>
          ) : (
            <div className="text-center text-slate-400">
              <Video className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p>Upload details and video to see the magic happen.</p>
              <p className="text-sm mt-2">Resulting video will appear here.</p>
            </div>
          )}
        </div>

      </main>
    </div>
  );
}