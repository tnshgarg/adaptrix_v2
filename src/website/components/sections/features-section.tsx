'use client'

import { motion } from 'framer-motion'
import { 
  Brain, 
  Zap, 
  Shield, 
  Globe, 
  Code, 
  FileText, 
  Calculator, 
  Scale,
  Layers,
  Target,
  Database,
  Cpu
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

export function FeaturesSection() {
  const features = [
    {
      icon: Brain,
      title: "MoE-LoRA System",
      description: "Automatic task classification with 100% accuracy across 4 domains. Intelligent adapter selection with confidence scoring.",
      details: ["Code Generation", "Legal Analysis", "General Knowledge", "Mathematical Reasoning"],
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: Layers,
      title: "Dynamic Adapter Composition",
      description: "Seamlessly combine multiple LoRA adapters for enhanced capabilities. Plug-and-play architecture for easy customization.",
      details: ["Universal Compatibility", "Hot-swapping", "Layer Injection", "Performance Tracking"],
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: Database,
      title: "RAG Integration",
      description: "FAISS vector store with intelligent document retrieval. Context-aware generation with source attribution.",
      details: ["Smart Chunking", "Semantic Search", "Reranking", "Source Attribution"],
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: Zap,
      title: "Optimized Inference",
      description: "vLLM integration with quantization support. Multi-level caching for lightning-fast responses.",
      details: ["vLLM Engine", "4-bit Quantization", "Batch Processing", "Response Caching"],
      color: "from-orange-500 to-red-500"
    },
    {
      icon: Globe,
      title: "REST API",
      description: "Production-ready FastAPI with comprehensive endpoints. Authentication, rate limiting, and real-time metrics.",
      details: ["OpenAPI Docs", "Rate Limiting", "Authentication", "Real-time Metrics"],
      color: "from-indigo-500 to-blue-500"
    },
    {
      icon: Shield,
      title: "Enterprise Ready",
      description: "Built for scale with security, monitoring, and compliance features. Deploy anywhere with confidence.",
      details: ["Security First", "Monitoring", "Compliance", "Scalability"],
      color: "from-gray-500 to-slate-500"
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
    <section id="features" className="py-24 bg-muted/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-4">
            Powerful Features for
            <span className="gradient-text"> Modern AI</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Adaptrix combines cutting-edge AI technologies to deliver unprecedented 
            modularity, performance, and ease of use.
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          {features.map((feature, index) => (
            <motion.div key={index} variants={itemVariants}>
              <Card className="h-full group hover:shadow-lg transition-all duration-300 border-0 bg-white/50 dark:bg-gray-800/50 backdrop-blur-sm">
                <CardHeader>
                  <div className="flex items-center space-x-4">
                    <div className={`p-3 rounded-lg bg-gradient-to-r ${feature.color} group-hover:scale-110 transition-transform duration-300`}>
                      <feature.icon className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-xl group-hover:text-primary transition-colors">
                        {feature.title}
                      </CardTitle>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <CardDescription className="text-base leading-relaxed">
                    {feature.description}
                  </CardDescription>
                  
                  <div className="space-y-2">
                    {feature.details.map((detail, detailIndex) => (
                      <div key={detailIndex} className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${feature.color}`} />
                        <span className="text-sm text-muted-foreground">{detail}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </motion.div>

        {/* Architecture Diagram */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
          className="mt-24"
        >
          <div className="text-center mb-12">
            <h3 className="text-2xl md:text-3xl font-bold mb-4">
              System Architecture
            </h3>
            <p className="text-lg text-muted-foreground">
              A comprehensive view of how Adaptrix components work together
            </p>
          </div>

          <div className="relative">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Input Layer */}
              <Card className="text-center">
                <CardHeader>
                  <Target className="w-8 h-8 mx-auto text-blue-600 mb-2" />
                  <CardTitle>Input Processing</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="text-sm space-y-1 text-muted-foreground">
                    <li>Task Classification</li>
                    <li>Adapter Selection</li>
                    <li>Context Retrieval</li>
                  </ul>
                </CardContent>
              </Card>

              {/* Processing Layer */}
              <Card className="text-center">
                <CardHeader>
                  <Cpu className="w-8 h-8 mx-auto text-purple-600 mb-2" />
                  <CardTitle>AI Processing</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="text-sm space-y-1 text-muted-foreground">
                    <li>Qwen-3 1.7B Model</li>
                    <li>LoRA Composition</li>
                    <li>vLLM Optimization</li>
                  </ul>
                </CardContent>
              </Card>

              {/* Output Layer */}
              <Card className="text-center">
                <CardHeader>
                  <Globe className="w-8 h-8 mx-auto text-green-600 mb-2" />
                  <CardTitle>Output Delivery</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="text-sm space-y-1 text-muted-foreground">
                    <li>Response Generation</li>
                    <li>Source Attribution</li>
                    <li>API Delivery</li>
                  </ul>
                </CardContent>
              </Card>
            </div>

            {/* Connection Lines */}
            <div className="hidden md:block absolute top-1/2 left-1/3 w-1/3 h-0.5 bg-gradient-to-r from-blue-600 to-purple-600 transform -translate-y-1/2" />
            <div className="hidden md:block absolute top-1/2 right-1/3 w-1/3 h-0.5 bg-gradient-to-r from-purple-600 to-green-600 transform -translate-y-1/2" />
          </div>
        </motion.div>
      </div>
    </section>
  )
}
