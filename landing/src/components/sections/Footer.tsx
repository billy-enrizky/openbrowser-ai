import Image from "next/image";

export function Footer() {
  return (
    <footer className="border-t border-white/5 bg-zinc-950 py-12 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Top row */}
        <div className="flex flex-col md:flex-row justify-between gap-8">
          {/* Left column: brand */}
          <div>
            <div className="flex items-center gap-2">
              <Image
                src="/logo.svg"
                alt="OpenBrowser"
                width={28}
                height={28}
              />
              <span className="font-bold text-lg text-white">OpenBrowser</span>
            </div>
            <p className="text-sm text-slate-500 mt-2">
              The general-purpose agentic browser.
            </p>
          </div>

          {/* Link columns */}
          <div className="flex gap-16">
            {/* Product */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-3">Product</h4>
              <a
                href="https://docs.openbrowser.me"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-slate-400 hover:text-white transition block mb-2"
              >
                Documentation
              </a>
              <a
                href="https://github.com/billy-enrizky/openbrowser-ai"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-slate-400 hover:text-white transition block mb-2"
              >
                GitHub
              </a>
              <a
                href="https://discord.gg/YRXzbJjq9K"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-slate-400 hover:text-white transition block mb-2"
              >
                Discord
              </a>
            </div>

            {/* Connect */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-3">Connect</h4>
              <a
                href="https://www.linkedin.com/in/enrizky-brillian/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-slate-400 hover:text-white transition block mb-2"
              >
                LinkedIn
              </a>
              <a
                href="https://github.com/billy-enrizky/openbrowser-ai"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-slate-400 hover:text-white transition block mb-2"
              >
                GitHub
              </a>
            </div>
          </div>
        </div>

        {/* Bottom row */}
        <div className="border-t border-white/5 mt-8 pt-8 flex flex-col md:flex-row justify-between text-sm text-slate-500">
          <span>2026 OpenBrowser. All rights reserved.</span>
          <span className="mt-2 md:mt-0">Open source under MIT license.</span>
        </div>
      </div>
    </footer>
  );
}
