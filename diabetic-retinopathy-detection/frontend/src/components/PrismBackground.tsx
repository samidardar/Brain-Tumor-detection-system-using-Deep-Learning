import { motion } from "framer-motion";

const PrismBackground = () => {
    return (
        <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10 bg-pink-light/30">
            {/* Prism 1 - Gold */}
            <motion.div
                className="absolute top-1/4 left-1/4 w-96 h-96 bg-gold/20 rounded-full blur-[100px]"
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
            {/* Prism 2 - Olive */}
            <motion.div
                className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-olive/20 rounded-full blur-[100px]"
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
            {/* Triangle Shape - Pink Accent */}
            <motion.div
                className="absolute top-1/2 left-1/2 w-64 h-64 border-t-[100px] border-l-[50px] border-r-[50px] border-b-[0px] border-transparent border-t-pink-DEFAULT/40 blur-xl transform -translate-x-1/2 -translate-y-1/2"
                animate={{ rotate: 360 }}
                transition={{ duration: 50, repeat: Infinity, ease: "linear" }}
            />

            {/* Grid Overlay */}
            <div
                className="absolute inset-0 opacity-10"
                style={{
                    backgroundImage: "radial-gradient(#808000 1px, transparent 1px)",
                    backgroundSize: "30px 30px"
                }}
            />
        </div>
    );
};

export default PrismBackground;
