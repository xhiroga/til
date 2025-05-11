import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export", // FastAPI等でホストする場合の設定。静的サイトをビルド。
  /* config options here */
};

export default nextConfig;
