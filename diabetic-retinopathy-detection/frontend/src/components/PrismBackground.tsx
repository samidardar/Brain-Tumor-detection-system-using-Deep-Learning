import { motion } from "framer-motion";

const PrismBackground = () => {
    return (
        <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10 bg-white">
            {/* Prism 1 - Primary Pink */}
            <motion.div
                className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-primary/10 rounded-full blur-[120px]"
                animate={{
                    x: [0, 100, 0],
                    y: [0, -50, 0],
                    scale: [1, 1.2, 1],
                }}
                transition={{
                    duration: 20,
                    repeat: Infinity,
                    repeatType: "reverse",
                }}
            />
            {/* Prism 2 - Secondary Green */}
            <motion.div
                className="absolute bottom-1/4 right-1/4 w-[600px] h-[600px] bg-secondary/10 rounded-full blur-[140px]"
                animate={{
                    x: [0, -100, 0],
                    y: [0, 50, 0],
                    scale: [1, 1.5, 1],
                }}
                transition={{
                    duration: 25,
                    repeat: Infinity,
                    repeatType: "reverse",
                }}
            />

            {/* Soft White Glow Center */}
            <div className="absolute inset-0 bg-white/40 blur-3xl -z-10" />

            {/* Grid Overlay - Very Subtle */}
            <div
                className="absolute inset-0 opacity-[0.03]"
                style={{
                    backgroundImage: "radial-gradient(#000 1px, transparent 1px)",
                    backgroundSize: "40px 40px"
                }}
            />
        </div>
    );
};

export default PrismBackground;
