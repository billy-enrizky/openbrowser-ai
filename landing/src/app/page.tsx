import { Navbar } from "@/components/sections/Navbar";
import { Hero } from "@/components/sections/Hero";
import { Features } from "@/components/sections/Features";

export default function Home() {
  return (
    <div className="bg-zinc-950 text-white">
      <Navbar />
      <Hero />
      <Features />
    </div>
  );
}
