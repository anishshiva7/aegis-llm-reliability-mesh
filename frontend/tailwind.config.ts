import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Slate-based "infra console" palette.
        ink: {
          950: "#070b14",
          900: "#0b1220",
          850: "#0f1729",
          800: "#131c30",
          700: "#1c2940",
          600: "#2a3a57",
        },
        accent: {
          DEFAULT: "#5b8cff",
          soft: "#8aa9ff",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 8px 30px -12px rgba(0,0,0,0.6)",
      },
    },
  },
  plugins: [],
};

export default config;
