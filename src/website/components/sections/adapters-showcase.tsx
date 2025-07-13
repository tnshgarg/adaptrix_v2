'use client'

import { motion } from 'framer-motion'
import { Code, Scale, Calculator, Brain, Star, Download, ArrowRight } from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

export function AdaptersShowcase() {
  const featuredAdapters = [
    {
      id: 'code',
      name: 'Code Generator Pro',
      description: 'Advanced code generation with debugging, optimization, and multi-language support.',
      icon: Code,
      color: 'from-blue-500 to-cyan-500',
      rating: 4.8,
      downloads: '15.4K',
      features: ['Python', 'JavaScript', 'TypeScript', 'Debugging', 'Optimization'],
      price: '$29.99'
    },
    {
      id: 'legal',
      name: 'Legal Analyzer Pro',
      description: 'Professional legal document analysis with contract review and compliance checking.',
      icon: Scale,
      color: 'from-purple-500 to-pink-500',
      rating: 4.9,
      downloads: '8.7K',
      features: ['Contract Analysis', 'Compliance', 'Risk Assessment', 'Legal Research'],
      price: '$49.99'
    },
    {
      id: 'math',
      name: 'Math Reasoning Engine',
      description: 'Advanced mathematical problem solving with step-by-step explanations.',
      icon: Calculator,
      color: 'from-green-500 to-emerald-500',
      rating: 4.7,
      downloads: '12.3K',
      features: ['Calculus', 'Algebra', 'Statistics', 'Graph Plotting'],
      price: '$19.99'
    },
    {
      id: 'creative',
      name: 'Creative Writing Assistant',
      description: 'Enhance your creative writing with style suggestions and plot development.',
      icon: Brain,
      color: 'from-orange-500 to-red-500',
      rating: 4.6,
      downloads: '9.8K',
      features: ['Style Enhancement', 'Plot Development', 'Character Creation'],
      price: '$24.99'
    }
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5
      }
    }
  }

  return (
    <section id="adapters" className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-4">
            Featured <span className="gradient-text">Adapters</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Discover specialized AI adapters that enhance your Adaptrix system 
            with domain-specific capabilities and expert knowledge.
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12"
        >
          {featuredAdapters.map((adapter) => (
            <motion.div key={adapter.id} variants={itemVariants}>
              <Card className="h-full group hover:shadow-xl transition-all duration-300 border-0 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className={`p-3 rounded-lg bg-gradient-to-r ${adapter.color} group-hover:scale-110 transition-transform duration-300`}>
                        <adapter.icon className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <CardTitle className="text-xl group-hover:text-primary transition-colors">
                          {adapter.name}
                        </CardTitle>
                        <div className="flex items-center space-x-4 mt-1">
                          <div className="flex items-center space-x-1">
                            <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                            <span className="text-sm text-muted-foreground">{adapter.rating}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Download className="w-4 h-4 text-muted-foreground" />
                            <span className="text-sm text-muted-foreground">{adapter.downloads}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold">{adapter.price}</div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <CardDescription className="text-base leading-relaxed">
                    {adapter.description}
                  </CardDescription>
                  
                  <div className="flex flex-wrap gap-2">
                    {adapter.features.map((feature, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {feature}
                      </Badge>
                    ))}
                  </div>

                  <div className="flex items-center justify-between pt-4">
                    <Link href={`/marketplace/${adapter.id}`}>
                      <Button variant="outline" size="sm">
                        View Details
                      </Button>
                    </Link>
                    <Link href="/marketplace">
                      <Button size="sm" className="group">
                        Add to Cart
                        <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
          className="text-center"
        >
          <Link href="/marketplace">
            <Button size="xl" variant="gradient" className="group">
              Explore All Adapters
              <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>
          <p className="text-sm text-muted-foreground mt-4">
            Over 50+ specialized adapters available in our marketplace
          </p>
        </motion.div>
      </div>
    </section>
  )
}
