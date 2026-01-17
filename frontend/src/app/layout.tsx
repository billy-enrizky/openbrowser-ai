import type { Metadata } from "next";
import "./globals.css";

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
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        {children}
      </body>
    </html>
  );
}
