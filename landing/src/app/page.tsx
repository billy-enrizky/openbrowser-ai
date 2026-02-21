import { Navbar } from "@/components/sections/Navbar";

export default function Home() {
  return (
    <main className="bg-zinc-950 text-white min-h-screen">
      <Navbar />
      <div className="flex items-center justify-center min-h-screen">
        <h1 className="text-4xl font-bold">OpenBrowser</h1>
      </div>
    </main>
  );
}
