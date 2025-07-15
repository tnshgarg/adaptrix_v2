'use client'

import { motion } from 'framer-motion'
import { ArrowRight, Sparkles, Zap, Brain } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SignUpButton, useUser } from '@clerk/nextjs'
import Link from 'next/link'

export function CTASection() {
  const { isSignedIn } = useUser()

  return (
    <section className="py-24 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0">
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 180, 360],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute top-1/4 left-1/4 w-64 h-64 bg-white/10 rounded-full blur-3xl"
        />
        <motion.div
          animate={{
            scale: [1.2, 1, 1.2],
            rotate: [360, 180, 0],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-white/10 rounded-full blur-3xl"
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="space-y-8"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            viewport={{ once: true }}
            className="inline-flex items-center px-4 py-2 rounded-full bg-white/20 backdrop-blur-sm border border-white/30"
          >
            <Sparkles className="w-4 h-4 text-white mr-2" />
            <span className="text-sm font-medium text-white">
              Join the AI Revolution
            </span>
          </motion.div>

          {/* Main Heading */}
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            viewport={{ once: true }}
            className="text-4xl md:text-5xl lg:text-6xl font-bold text-white"
          >
            Ready to Transform Your
            <br />
            <span className="text-yellow-300">AI Capabilities?</span>
          </motion.h2>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            viewport={{ once: true }}
            className="text-xl md:text-2xl text-white/90 max-w-3xl mx-auto leading-relaxed"
          >
            Start building with the world's first modular AI system today. 
            No setup required, instant access to powerful adapters.
          </motion.p>

          {/* Feature Pills */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            viewport={{ once: true }}
            className="flex flex-wrap justify-center gap-4 text-sm"
          >
            {[
              { icon: Brain, text: "100% Accurate Classification" },
              { icon: Zap, text: "Lightning Fast Responses" },
              { icon: ArrowRight, text: "Instant Setup" }
            ].map((feature, index) => (
              <div
                key={index}
                className="flex items-center px-4 py-2 rounded-full bg-white/20 backdrop-blur-sm border border-white/30"
              >
                <feature.icon className="w-4 h-4 mr-2 text-white" />
                <span className="font-medium text-white">{feature.text}</span>
              </div>
            ))}
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.8 }}
            viewport={{ once: true }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            {isSignedIn ? (
              <Link href="/dashboard">
                <Button 
                  size="xl" 
                  className="bg-white text-purple-600 hover:bg-white/90 group text-lg px-8 py-4"
                >
                  Go to Dashboard
                  <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
            ) : (
              <SignUpButton mode="modal">
                <Button 
                  size="xl" 
                  className="bg-white text-purple-600 hover:bg-white/90 group text-lg px-8 py-4"
                >
                  Start Building Now
                  <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                </Button>
              </SignUpButton>
            )}
            
            <Link href="/marketplace">
              <Button 
                size="xl" 
                variant="outline" 
                className="border-white/30 text-white hover:bg-white/10 text-lg px-8 py-4"
              >
                Explore Adapters
              </Button>
            </Link>
          </motion.div>

          {/* Trust Indicators */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.8 }}
            viewport={{ once: true }}
            className="pt-8 text-white/80"
          >
            <p className="text-sm mb-4">Trusted by developers at</p>
            <div className="flex flex-wrap justify-center items-center gap-8 opacity-60">
              {['TechCorp', 'CodeCraft', 'LegalAI', 'Analytics Pro', 'InnovateTech'].map((company) => (
                <div key={company} className="text-lg font-semibold">
                  {company}
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}
