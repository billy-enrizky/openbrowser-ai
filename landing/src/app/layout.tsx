import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OpenBrowser - The General-Purpose Agentic Browser",
  description:
    "Control any browser with natural language. 6x fewer tokens than Chrome DevTools MCP. 144x fewer response tokens. Open source.",
  openGraph: {
    title: "OpenBrowser - The General-Purpose Agentic Browser",
    description:
      "Control any browser with natural language. 6x fewer tokens than Chrome DevTools MCP. 144x fewer response tokens. Open source.",
    url: "https://openbrowser.me",
    siteName: "OpenBrowser",
    type: "website",
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
      <body
        className={`${geistSans.variable} ${geistMono.variable} bg-background text-foreground antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
