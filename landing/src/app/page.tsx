import { Navbar } from "@/components/sections/Navbar";
import { Hero } from "@/components/sections/Hero";
import { Features } from "@/components/sections/Features";
import { HowItWorks } from "@/components/sections/HowItWorks";
import { VideoDemo } from "@/components/sections/VideoDemo";
import { Integrations } from "@/components/sections/Integrations";
import { Waitlist } from "@/components/sections/Waitlist";
import { Footer } from "@/components/sections/Footer";

function SectionDivider() {
  return (
    <div className="mx-auto max-w-5xl px-6">
      <div className="h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />
    </div>
  );
}

export default function Home() {
  return (
    <div className="bg-zinc-950 text-white">
      <Navbar />
      <Hero />
      <SectionDivider />
      <Features />
      <SectionDivider />
      <HowItWorks />
      <SectionDivider />
      <VideoDemo />
      <SectionDivider />
      <Integrations />
      <SectionDivider />
      <Waitlist />
      <Footer />
    </div>
  );
}
