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
    gradcam_base64: string; // base64
    referable: boolean;
}

const FileUpload = () => {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<PredictionResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const [patientName, setPatientName] = useState("");
    const [patientId, setPatientId] = useState("");

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
            console.log("API Response:", data); // Debug log
            setResult(data);
        } catch (err: any) {
            setError(err.message || "An unexpected error occurred.");
        } finally {
            setLoading(false);
        }
    };

    const DIAGNOSIS_DETAILS: Record<number, { description: string; action: string; followUp: string }> = {
        0: {
            description: "No apparent retinopathic abnormalities detected. The retina appears healthy.",
            action: "Maintain routine annual screening.",
            followUp: "12 Months"
        },
        1: {
            description: "Mild Non-Proliferative Diabetic Retinopathy (NPDR). Presence of microaneurysms only.",
            action: "Manage diabetes and hypertension strictly. Rescreen to monitor progression.",
            followUp: "6-12 Months"
        },
        2: {
            description: "Moderate Non-Proliferative Diabetic Retinopathy (NPDR). More than just microaneurysms but less than Severe NPDR.",
            action: "Referral to ophthalmologist for evaluation. STRICT glycemic control recommended.",
            followUp: "3-6 Months"
        },
        3: {
            description: "Severe Non-Proliferative Diabetic Retinopathy (NPDR). Signs include >20 intraretinal hemorrhages, venous beading, or IRMA.",
            action: "Urgent referral to ophthalmologist. Potential need for panretinal photocoagulation.",
            followUp: "2-4 Weeks"
        },
        4: {
            description: "Proliferative Diabetic Retinopathy (PDR). Presence of neovascularization or vitreous hemorrhage.",
            action: "EMERGENCY referral. High risk of vision loss. Likely requires anti-VEGF or surgery.",
            followUp: "Immediate"
        }
    };

    // For Binary Model (0=No Referral, 1=Referral) - approximate mapping if grade is just 0/1
    const BINARY_DETAILS: Record<number, { description: string; action: string; followUp: string }> = {
        0: DIAGNOSIS_DETAILS[0],
        1: {
            description: "Signs of Referable Diabetic Retinopathy detected (Moderate NPDR or worse).",
            action: "Referral to ophthalmologist required for full dilated exam and grading.",
            followUp: "As soon as possible"
        }
    };

    return (
        <div className="w-full max-w-5xl mx-auto p-4 md:p-8 space-y-8">
            {/* Header / Patient Info Input */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white/90 backdrop-blur-md rounded-3xl shadow-sm border border-gray-100 p-6 flex flex-col md:flex-row gap-4 items-end"
            >
                <div className="flex-1 w-full">
                    <label className="text-xs font-semibold uppercase text-gray-400 tracking-wider ml-1">Patient Name</label>
                    <input
                        type="text"
                        placeholder="e.g. John Doe"
                        value={patientName}
                        onChange={(e) => setPatientName(e.target.value)}
                        className="w-full mt-1 p-3 bg-gray-50 border-gray-200 rounded-xl focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none transition-all"
                    />
                </div>
                <div className="flex-1 w-full">
                    <label className="text-xs font-semibold uppercase text-gray-400 tracking-wider ml-1">Patient ID / MRN</label>
                    <input
                        type="text"
                        placeholder="e.g. MRN-2024-8892"
                        value={patientId}
                        onChange={(e) => setPatientId(e.target.value)}
                        className="w-full mt-1 p-3 bg-gray-50 border-gray-200 rounded-xl focus:ring-2 focus:ring-primary/20 focus:border-primary outline-none transition-all"
                    />
                </div>
            </motion.div>

            {/* Upload Zone */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white/90 backdrop-blur-md rounded-3xl shadow-xl overflow-hidden border border-gray-100"
            >
                <div className="p-8">
                    <div
                        {...getRootProps()}
                        className={`border-3 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${isDragActive
                            ? "border-primary bg-primary/5 scale-[1.02]"
                            : "border-gray-200 hover:border-primary/50 hover:bg-gray-50"
                            }`}
                    >
                        <input {...getInputProps()} />
                        <div className="flex flex-col items-center gap-4">
                            <div className="p-4 bg-primary/10 rounded-full text-primary">
                                <Upload className="w-8 h-8" />
                            </div>
                            <div>
                                <h3 className="text-xl font-semibold text-gray-900">
                                    {isDragActive ? "Drop fundus image here" : "Upload Retinal Scan"}
                                </h3>
                                <p className="text-gray-500 mt-2">
                                    Drag & drop or click to select a fundus image
                                </p>
                                <p className="text-xs text-gray-400 mt-4 font-mono">
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
                                className="bg-primary hover:bg-primary/90 text-white font-bold px-8 shadow-lg shadow-primary/20 transition-all rounded-full"
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
                        className="space-y-6"
                    >
                        <div className="grid md:grid-cols-2 gap-6">
                            {/* Diagnosis Card */}
                            <div className={`rounded-3xl p-8 text-white shadow-2xl relative overflow-hidden ${result.referable ? "bg-gradient-to-br from-red-500 to-primary" : "bg-gradient-to-br from-secondary to-teal-600"
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

                                    <div className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-md px-4 py-2 rounded-full text-sm font-medium border border-white/30">
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
                                    <h3 className="font-semibold text-gray-900 border-l-4 border-primary pl-3">Visual Explanation</h3>
                                    <div className="flex gap-2">
                                        <span className="text-[10px] uppercase font-bold text-gray-400 bg-gray-50 px-2 py-1 rounded-md border border-gray-100">Grad-CAM</span>
                                    </div>
                                </div>

                                <div className="flex-1 rounded-xl overflow-hidden border border-gray-100 bg-gray-50 relative group">
                                    {result.gradcam_base64 ? (
                                        <img
                                            src={`data:image/png;base64,${result.gradcam_base64}`}
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
                        </div>

                        {/* Detailed Medical Report */}
                        <div className="bg-white rounded-3xl p-8 shadow-lg border border-gray-100">
                            <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                                <Activity className="w-5 h-5 text-secondary" />
                                Clinical Analysis Report
                            </h3>

                            <div className="grid md:grid-cols-2 gap-8">
                                <div className="space-y-4">
                                    <div>
                                        <label className="text-xs font-semibold uppercase text-gray-400 tracking-wider">Patient</label>
                                        <p className="text-lg font-medium text-gray-900">{patientName || "Anonymous"}</p>
                                        <p className="text-sm text-gray-500 font-mono">{patientId || "ID: N/A"}</p>
                                    </div>
                                    <div className="pt-4 border-t border-gray-100">
                                        <label className="text-xs font-semibold uppercase text-gray-400 tracking-wider">Clinical Finding</label>
                                        <p className="text-gray-700 mt-1 leading-relaxed">
                                            {(result.grade in DIAGNOSIS_DETAILS)
                                                ? DIAGNOSIS_DETAILS[result.grade].description
                                                : (result.referable ? BINARY_DETAILS[1].description : BINARY_DETAILS[0].description)}
                                        </p>
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <div className="bg-secondary/5 rounded-2xl p-6 border border-secondary/10">
                                        <label className="text-xs font-semibold uppercase text-secondary/80 tracking-wider flex items-center gap-2">
                                            <CheckCircle className="w-3 h-3" /> Recommended Action
                                        </label>
                                        <p className="text-secondary font-medium mt-2 text-lg">
                                            {(result.grade in DIAGNOSIS_DETAILS)
                                                ? DIAGNOSIS_DETAILS[result.grade].action
                                                : (result.referable ? BINARY_DETAILS[1].action : BINARY_DETAILS[0].action)}
                                        </p>
                                    </div>

                                    <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-xl">
                                        <div className="bg-white p-2 rounded-lg shadow-sm font-bold text-gray-900 border border-gray-100">
                                            {(result.grade in DIAGNOSIS_DETAILS)
                                                ? DIAGNOSIS_DETAILS[result.grade].followUp
                                                : (result.referable ? BINARY_DETAILS[1].followUp : BINARY_DETAILS[0].followUp)}
                                        </div>
                                        <span className="text-sm text-gray-500 font-medium">Recommended Follow-up Interval</span>
                                    </div>
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
