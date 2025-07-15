'use client'

import { motion } from 'framer-motion'
import { Zap, Database, Brain, Shield, Globe, Cpu } from 'lucide-react'

export function TechStack() {
  const technologies = [
    {
      category: 'AI Models',
      icon: Brain,
      items: [
        { name: 'Qwen-3 1.7B', description: 'Primary base model' },
        { name: 'LoRA Adapters', description: 'Modular capabilities' },
        { name: 'MoE Classification', description: '100% accuracy' }
      ]
    },
    {
      category: 'Performance',
      icon: Zap,
      items: [
        { name: 'vLLM Engine', description: 'High-throughput inference' },
        { name: '4-bit Quantization', description: 'Memory optimization' },
        { name: 'Multi-level Caching', description: 'Response acceleration' }
      ]
    },
    {
      category: 'Data & Retrieval',
      icon: Database,
      items: [
        { name: 'FAISS Vector Store', description: 'Semantic search' },
        { name: 'RAG Pipeline', description: 'Context augmentation' },
        { name: 'Smart Chunking', description: 'Document processing' }
      ]
    },
    {
      category: 'Infrastructure',
      icon: Cpu,
      items: [
        { name: 'FastAPI', description: 'Modern web framework' },
        { name: 'PyTorch', description: 'Deep learning backend' },
        { name: 'Docker', description: 'Containerization' }
      ]
    },
    {
      category: 'Security',
      icon: Shield,
      items: [
        { name: 'Authentication', description: 'Secure access control' },
        { name: 'Rate Limiting', description: 'Usage protection' },
        { name: 'API Keys', description: 'Secure integration' }
      ]
    },
    {
      category: 'API & Integration',
      icon: Globe,
      items: [
        { name: 'REST API', description: 'Standard integration' },
        { name: 'OpenAPI Docs', description: 'Auto-generated docs' },
        { name: 'SDKs', description: 'Multiple languages' }
      ]
    }
  ]

  return (
    <section className="py-24 bg-muted/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-4">
            Built with <span className="gradient-text">Cutting-Edge</span> Technology
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Adaptrix leverages the latest advances in AI, performance optimization, 
            and cloud infrastructure to deliver unmatched capabilities.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {technologies.map((tech, index) => (
            <motion.div
              key={tech.category}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="bg-background rounded-lg p-6 shadow-sm border"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <tech.icon className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-lg font-semibold">{tech.category}</h3>
              </div>
              
              <div className="space-y-3">
                {tech.items.map((item, itemIndex) => (
                  <div key={itemIndex} className="flex justify-between items-start">
                    <div>
                      <div className="font-medium text-sm">{item.name}</div>
                      <div className="text-xs text-muted-foreground">{item.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Performance Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
          className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8 text-center"
        >
          <div>
            <div className="text-3xl font-bold gradient-text">96.5%</div>
            <div className="text-sm text-muted-foreground">System Accuracy</div>
          </div>
          <div>
            <div className="text-3xl font-bold gradient-text">&lt;200ms</div>
            <div className="text-sm text-muted-foreground">API Response</div>
          </div>
          <div>
            <div className="text-3xl font-bold gradient-text">32K</div>
            <div className="text-sm text-muted-foreground">Context Length</div>
          </div>
          <div>
            <div className="text-3xl font-bold gradient-text">99.9%</div>
            <div className="text-sm text-muted-foreground">Uptime SLA</div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
