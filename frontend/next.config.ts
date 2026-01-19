import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable static export for GitHub Pages
  output: "export",
  
  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
  
  // Trailing slash for GitHub Pages compatibility
  trailingSlash: true,
  
  // Transpile react-vnc package for proper ESM support
  transpilePackages: ["react-vnc"],
};

export default nextConfig;
