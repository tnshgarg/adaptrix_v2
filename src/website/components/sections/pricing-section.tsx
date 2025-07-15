'use client'

import { motion } from 'framer-motion'
import { Check, Star, Zap, Crown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { SignUpButton, useUser } from '@clerk/nextjs'
import Link from 'next/link'

export function PricingSection() {
  const { isSignedIn } = useUser()

  const plans = [
    {
      name: 'Free',
      description: 'Perfect for getting started',
      price: '$0',
      period: 'forever',
      icon: Star,
      features: [
        '1,000 API calls/month',
        'Basic adapters included',
        'Community support',
        'Standard response time',
        'Basic documentation'
      ],
      limitations: [
        'No custom adapters',
        'Limited RAG documents',
        'No priority support'
      ],
      cta: 'Get Started',
      popular: false
    },
    {
      name: 'Pro',
      description: 'For growing teams and projects',
      price: '$29',
      period: 'per month',
      icon: Zap,
      features: [
        '50,000 API calls/month',
        'All premium adapters',
        'Priority support',
        'Advanced RAG features',
        'Custom adapter uploads',
        'Team collaboration',
        'Advanced analytics',
        'API rate limiting controls'
      ],
      limitations: [],
      cta: 'Start Pro Trial',
      popular: true
    },
    {
      name: 'Enterprise',
      description: 'For large organizations',
      price: 'Custom',
      period: 'contact us',
      icon: Crown,
      features: [
        'Unlimited API calls',
        'Custom adapter development',
        'Dedicated support team',
        'On-premise deployment',
        'Custom integrations',
        'SLA guarantees',
        'Advanced security',
        'Training & onboarding'
      ],
      limitations: [],
      cta: 'Contact Sales',
      popular: false
    }
  ]

  return (
    <section id="pricing" className="py-24 bg-muted/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-4">
            Simple, <span className="gradient-text">Transparent</span> Pricing
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Choose the plan that fits your needs. Start free and scale as you grow.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="relative"
            >
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <Badge className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-1">
                    Most Popular
                  </Badge>
                </div>
              )}
              
              <Card className={`h-full ${plan.popular ? 'ring-2 ring-blue-600 shadow-xl' : ''}`}>
                <CardHeader className="text-center pb-8">
                  <div className="flex justify-center mb-4">
                    <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                      plan.popular 
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600' 
                        : 'bg-muted'
                    }`}>
                      <plan.icon className={`w-6 h-6 ${plan.popular ? 'text-white' : 'text-muted-foreground'}`} />
                    </div>
                  </div>
                  
                  <CardTitle className="text-2xl">{plan.name}</CardTitle>
                  <CardDescription className="text-base">{plan.description}</CardDescription>
                  
                  <div className="mt-4">
                    <div className="text-4xl font-bold">{plan.price}</div>
                    <div className="text-muted-foreground">{plan.period}</div>
                  </div>
                </CardHeader>

                <CardContent className="space-y-6">
                  {/* Features */}
                  <div className="space-y-3">
                    {plan.features.map((feature, featureIndex) => (
                      <div key={featureIndex} className="flex items-center space-x-3">
                        <Check className="w-4 h-4 text-green-600 shrink-0" />
                        <span className="text-sm">{feature}</span>
                      </div>
                    ))}
                  </div>

                  {/* CTA Button */}
                  <div className="pt-6">
                    {plan.name === 'Enterprise' ? (
                      <Button 
                        className="w-full" 
                        variant={plan.popular ? "default" : "outline"}
                        size="lg"
                      >
                        {plan.cta}
                      </Button>
                    ) : isSignedIn ? (
                      <Link href="/dashboard">
                        <Button 
                          className="w-full" 
                          variant={plan.popular ? "default" : "outline"}
                          size="lg"
                        >
                          Go to Dashboard
                        </Button>
                      </Link>
                    ) : (
                      <SignUpButton mode="modal">
                        <Button 
                          className="w-full" 
                          variant={plan.popular ? "default" : "outline"}
                          size="lg"
                        >
                          {plan.cta}
                        </Button>
                      </SignUpButton>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* FAQ */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <h3 className="text-2xl font-bold mb-8">Frequently Asked Questions</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-left max-w-4xl mx-auto">
            <div>
              <h4 className="font-semibold mb-2">Can I change plans anytime?</h4>
              <p className="text-muted-foreground text-sm">
                Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">What happens if I exceed my limits?</h4>
              <p className="text-muted-foreground text-sm">
                We'll notify you when you're approaching your limits. You can upgrade or purchase additional credits.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Do you offer refunds?</h4>
              <p className="text-muted-foreground text-sm">
                Yes, we offer a 30-day money-back guarantee for all paid plans.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-2">Is there a free trial?</h4>
              <p className="text-muted-foreground text-sm">
                Yes, Pro plan includes a 14-day free trial. No credit card required to start.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
