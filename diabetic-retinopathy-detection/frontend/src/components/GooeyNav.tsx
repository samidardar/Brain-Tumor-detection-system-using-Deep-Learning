import { useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const tabs = [
    { id: "home", label: "Home" },
    { id: "analyze", label: "Analyze Fundus" },
    { id: "history", label: "History" },
    { id: "settings", label: "Settings" },
];

const GooeyNav = () => {
    const [activeTab, setActiveTab] = useState(tabs[1].id);

    return (
        <nav className="fixed top-6 left-1/2 -translate-x-1/2 z-50">
            <div className="flex p-1 bg-white/80 backdrop-blur-md rounded-full shadow-lg border border-white/20">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={cn(
                            "relative px-4 py-2 text-sm font-medium transition-colors duration-300 rounded-full",
                            activeTab === tab.id ? "text-olive font-bold" : "text-gray-500 hover:text-gray-900"
                        )}
                        style={{
                            WebkitTapHighlightColor: "transparent",
                        }}
                    >
                        {activeTab === tab.id && (
                            <motion.div
                                layoutId="gooey-pill"
                                className="absolute inset-0 bg-gold/30 rounded-full -z-10"
                                transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                            />
                        )}
                        {tab.label}
                    </button>
                ))}
            </div>
        </nav>
    );
};

export default GooeyNav;
