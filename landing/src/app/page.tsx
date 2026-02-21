import { Navbar } from "@/components/sections/Navbar";
import { Hero } from "@/components/sections/Hero";
import { Features } from "@/components/sections/Features";
import { HowItWorks } from "@/components/sections/HowItWorks";
import { VideoDemo } from "@/components/sections/VideoDemo";
import { Integrations } from "@/components/sections/Integrations";
import { Waitlist } from "@/components/sections/Waitlist";
import { Footer } from "@/components/sections/Footer";

export default function Home() {
  return (
    <div className="bg-zinc-950 text-white">
      <Navbar />
      <Hero />
      <Features />
      <HowItWorks />
      <VideoDemo />
      <Integrations />
      <Waitlist />
      <Footer />
    </div>
  );
}
