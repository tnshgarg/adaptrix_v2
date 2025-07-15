"use client";

import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";
import { HeroSection } from "@/components/sections/hero-section";
import { FeaturesSection } from "@/components/sections/features-section";
import { CLISection } from "@/components/sections/cli-section";
import { AdaptersShowcase } from "@/components/sections/adapters-showcase";
import { TechStack } from "@/components/sections/tech-stack";
import { Testimonials } from "@/components/sections/testimonials";
import { PricingSection } from "@/components/sections/pricing-section";
import { CTASection } from "@/components/sections/cta-section";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      {/* Hero Section */}
      <HeroSection />

      {/* Features Section */}
      <FeaturesSection />

      {/* CLI Section */}
      <CLISection />

      {/* Adapters Showcase */}
      <AdaptersShowcase />

      {/* Tech Stack */}
      <TechStack />

      {/* Testimonials */}
      <Testimonials />

      {/* Pricing */}
      <PricingSection />

      {/* CTA Section */}
      <CTASection />

      <Footer />
    </div>
  );
}
