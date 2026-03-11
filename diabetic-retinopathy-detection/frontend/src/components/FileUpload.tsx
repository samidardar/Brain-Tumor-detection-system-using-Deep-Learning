import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, CheckCircle, AlertCircle, Loader2, Eye, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";

interface PredictionResult {
    filename: string;
    grade: number;
    label: string;
    confidence: number;
    gradcam_image: string; // base64
    referable: boolean;
}

const FileUpload = () => {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        const selectedFile = acceptedFiles[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreview(URL.createObjectURL(selectedFile));
            setResult(null);
            setError(null);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { "image/*": [] },
        multiple: false,
    });

    const handleAnalyze = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            // Assuming backend is proxy-configured or CORS enabled at localhost:8000
            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Analysis failed. Please try again.");
            }

            const data = await response.json();
            setResult(data);
        } catch (err: any) {
            setError(err.message || "An unexpected error occurred.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-6 space-y-8">
            {/* Upload Zone */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl overflow-hidden border border-white/50"
            >
                <div className="p-8">
                    <div
                        {...getRootProps()}
                        className={`border-3 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${isDragActive
                                ? "border-gold bg-gold/5 scale-[1.02]"
                                : "border-gray-200 hover:border-gold/50 hover:bg-gray-50"
                            }`}
                    >
                        <input {...getInputProps()} />
                        <div className="flex flex-col items-center gap-4">
                            <div className="p-4 bg-pink-light/20 rounded-full text-olive">
                                <Upload className="w-8 h-8" />
                            </div>
                            <div>
                                <h3 className="text-xl font-semibold text-gray-900">
                                    {isDragActive ? "Drop fundus image here" : "Upload Retinal Scan"}
                                </h3>
                                <p className="text-gray-500 mt-2">
                                    Drag & drop or click to select a fundus image
                                </p>
                                <p className="text-xs text-olive/60 mt-4 font-mono">
                                    SUPPORTS: JPG, PNG • MAX 10MB
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Preview & Action */}
                <AnimatePresence>
                    {file && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="border-t border-gray-100 bg-gray-50/50 p-6 flex items-center justify-between"
                        >
                            <div className="flex items-center gap-4">
                                <div className="w-16 h-16 rounded-lg overflow-hidden border border-gray-200 shadow-sm relative group">
                                    <img src={preview!} alt="Preview" className="w-full h-full object-cover" />
                                </div>
                                <div>
                                    <p className="font-medium text-gray-900 truncate max-w-[200px]">{file.name}</p>
                                    <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                </div>
                            </div>

                            <Button
                                onClick={handleAnalyze}
                                disabled={loading}
                                className="bg-gold hover:bg-gold/90 text-olive font-bold px-8 shadow-lg shadow-gold/20"
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing...
                                    </>
                                ) : (
                                    <>
                                        <Activity className="w-4 h-4 mr-2" /> Start Analysis
                                    </>
                                )}
                            </Button>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>

            {/* Error Message */}
            {error && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="p-4 bg-red-50 text-red-600 rounded-xl flex items-center gap-3 border border-red-100"
                >
                    <AlertCircle className="w-5 h-5" />
                    {error}
                </motion.div>
            )}

            {/* Results Section */}
            <AnimatePresence>
                {result && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="grid md:grid-cols-2 gap-6"
                    >
                        {/* Diagnosis Card */}
                        <div className={`rounded-3xl p-8 text-white shadow-2xl relative overflow-hidden ${result.referable ? "bg-gradient-to-br from-red-500 to-pink-600" : "bg-gradient-to-br from-olive to-emerald-600"
                            }`}>
                            <div className="relative z-10">
                                <div className="flex items-center gap-3 mb-6 opacity-90">
                                    <Eye className="w-6 h-6" />
                                    <span className="font-mono tracking-wider text-sm uppercase">AI Diagnosis</span>
                                </div>

                                <h2 className="text-4xl font-bold mb-2">
                                    {result.label}
                                </h2>
                                <div className="text-lg opacity-90 mb-8 font-light">
                                    Grade {result.grade} • {(result.confidence * 100).toFixed(1)}% Confidence
                                </div>

                                <div className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-md px-4 py-2 rounded-full text-sm font-medium">
                                    {result.referable ? (
                                        <>
                                            <AlertCircle className="w-4 h-4" /> Referral Recommended
                                        </>
                                    ) : (
                                        <>
                                            <CheckCircle className="w-4 h-4" /> No Referral Needed
                                        </>
                                    )}
                                </div>
                            </div>

                            {/* Background Pattern */}
                            <div className="absolute right-0 bottom-0 opacity-10 transform translate-x-1/4 translate-y-1/4">
                                <Activity className="w-64 h-64" />
                            </div>
                        </div>

                        {/* Grad-CAM Visualization */}
                        <div className="bg-white rounded-3xl p-6 shadow-xl border border-gray-100 flex flex-col">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="font-semibold text-gray-900">Visual Explanation (Grad-CAM)</h3>
                                <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded">ResNet50 / EfficientNet</span>
                            </div>

                            <div className="flex-1 rounded-xl overflow-hidden border border-gray-100 bg-gray-50 relative group">
                                {result.gradcam_image ? (
                                    <img
                                        src={`data:image/png;base64,${result.gradcam_image}`}
                                        alt="Grad-CAM Heatmap"
                                        className="w-full h-full object-contain"
                                    />
                                ) : (
                                    <div className="flex items-center justify-center h-full text-gray-400">
                                        No visualization available
                                    </div>
                                )}

                                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white text-sm font-medium backdrop-blur-sm">
                                    Heatmap shows attention regions
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default FileUpload;
