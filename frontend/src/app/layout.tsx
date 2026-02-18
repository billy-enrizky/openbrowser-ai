import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/components/auth";

const EXTENSION_WS_URL = (process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws")
  .replace(/\/ws$/, "/ws/extension");

export const metadata: Metadata = {
  title: "OpenBrowser - AI Browser Automation",
  description: "Control your browser with natural language. OpenBrowser is an open-source AI browser automation framework.",
  keywords: ["AI", "browser automation", "web scraping", "AI agent", "OpenBrowser"],
  authors: [{ name: "Billy Enrizky" }],
  openGraph: {
    title: "OpenBrowser - AI Browser Automation",
    description: "Control your browser with natural language",
    url: "https://openbrowser.me",
    siteName: "OpenBrowser",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "OpenBrowser - AI Browser Automation",
    description: "Control your browser with natural language",
  },
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <meta name="openbrowser-ws-url" content={EXTENSION_WS_URL} />
      </head>
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
