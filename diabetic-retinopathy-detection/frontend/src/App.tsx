import PrismBackground from "./components/PrismBackground";
import GooeyNav from "./components/GooeyNav";
import FileUpload from "./components/FileUpload";
import { motion } from "framer-motion";

function App() {
  return (
    <div className="min-h-screen font-sans text-gray-900 selection:bg-gold/30">
      <PrismBackground />
      <GooeyNav />
      {/* Main Content */}
      <main className="container mx-auto px-4 pt-32 pb-16 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-center mb-16 space-y-4"
        >
          <div className="inline-block relative">
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-olive to-emerald-900 pb-2">
              Diabetic Retinopathy
            </h1>
            <div className="absolute -right-8 -top-8 text-6xl animate-bounce">👁️</div>
          </div>
          <h2 className="text-2xl md:text-3xl font-light text-gray-600">
            Advanced Clinical AI Screening System
          </h2>
          <p className="max-w-2xl mx-auto text-lg text-gray-500 leading-relaxed">
            Upload retinal fundus images for instant, high-precision analysis.
            Powered by state-of-the-art Deep Learning models with 97% AUC.
          </p>
        </motion.div>

        {/* Upload Interface */}
        <FileUpload />

      </main>

      {/* Footer */}
      <footer className="fixed bottom-4 w-full text-center text-xs text-gray-400 font-mono z-0 pointer-events-none">
        MEDICAL AI TEAM • v2.0 Production • <span className="text-olive">SECURE ENCLAVE</span>
      </footer>
    </div>
  );
}

export default App;
