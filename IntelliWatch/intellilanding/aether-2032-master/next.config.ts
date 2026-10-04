import type { NextConfig } from "next";

const isGitHubPages = process.env.GITHUB_PAGES === 'true';

const nextConfig: NextConfig = {
  // Silence workspace root warning
  turbopack: {
    root: __dirname,
  },

  // Static export
  output: 'export',

  // Required for static hosting
  images: {
    unoptimized: true,
  },

  // Trailing slashes (recommended for static hosting)
  trailingSlash: true,

  // When deploying to GitHub Pages (project site), we need basePath
  basePath: isGitHubPages ? '/aether-2032' : '',
  assetPrefix: isGitHubPages ? '/aether-2032/' : '',
};

export default nextConfig;
