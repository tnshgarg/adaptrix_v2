'use client'

import { motion } from 'framer-motion'
import { Star, Quote } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

export function Testimonials() {
  const testimonials = [
    {
      id: 1,
      name: 'Sarah Chen',
      role: 'AI Research Director',
      company: 'TechCorp',
      avatar: 'https://via.placeholder.com/64x64?text=SC',
      rating: 5,
      content: 'Adaptrix has revolutionized how we approach AI development. The modular adapter system allows us to quickly deploy specialized capabilities without retraining entire models. The 100% accuracy in task classification is remarkable.'
    },
    {
      id: 2,
      name: 'Marcus Rodriguez',
      role: 'Senior Developer',
      company: 'CodeCraft Solutions',
      avatar: 'https://via.placeholder.com/64x64?text=MR',
      rating: 5,
      content: 'The code generation adapter is incredibly powerful. It understands context, generates clean code, and even helps with debugging. The integration with our existing workflow was seamless.'
    },
    {
      id: 3,
      name: 'Dr. Emily Watson',
      role: 'Legal Tech Consultant',
      company: 'LegalAI Innovations',
      avatar: 'https://via.placeholder.com/64x64?text=EW',
      rating: 5,
      content: 'The legal document analysis capabilities are outstanding. Adaptrix helps us review contracts faster while maintaining accuracy. The RAG integration provides relevant precedents instantly.'
    },
    {
      id: 4,
      name: 'David Kim',
      role: 'Data Scientist',
      company: 'Analytics Pro',
      avatar: 'https://via.placeholder.com/64x64?text=DK',
      rating: 5,
      content: 'The mathematical reasoning engine is a game-changer for our data analysis workflows. Step-by-step explanations help our team understand complex calculations and validate results.'
    },
    {
      id: 5,
      name: 'Lisa Thompson',
      role: 'Product Manager',
      company: 'InnovateTech',
      avatar: 'https://via.placeholder.com/64x64?text=LT',
      rating: 5,
      content: 'Adaptrix\'s API is incredibly well-designed. The documentation is comprehensive, and the performance is exceptional. Our team was able to integrate it in just a few hours.'
    },
    {
      id: 6,
      name: 'James Wilson',
      role: 'CTO',
      company: 'StartupXYZ',
      avatar: 'https://via.placeholder.com/64x64?text=JW',
      rating: 5,
      content: 'The modular approach of Adaptrix allows us to scale our AI capabilities as we grow. We started with one adapter and now use multiple specialized ones. The cost efficiency is remarkable.'
    }
  ]

  return (
    <section className="py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold mb-4">
            Trusted by <span className="gradient-text">Developers</span> Worldwide
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            See what our users are saying about their experience with Adaptrix
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={testimonial.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              <Card className="h-full hover:shadow-lg transition-all duration-300">
                <CardContent className="p-6">
                  {/* Quote Icon */}
                  <Quote className="w-8 h-8 text-muted-foreground mb-4" />
                  
                  {/* Rating */}
                  <div className="flex items-center space-x-1 mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                    ))}
                  </div>

                  {/* Content */}
                  <p className="text-muted-foreground mb-6 leading-relaxed">
                    "{testimonial.content}"
                  </p>

                  {/* Author */}
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-600 to-purple-600 flex items-center justify-center text-white font-semibold">
                      {testimonial.name.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div>
                      <div className="font-semibold">{testimonial.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {testimonial.role} at {testimonial.company}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
          className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8 text-center"
        >
          <div>
            <div className="text-3xl font-bold gradient-text">10,000+</div>
            <div className="text-sm text-muted-foreground">Active Users</div>
          </div>
          <div>
            <div className="text-3xl font-bold gradient-text">50+</div>
            <div className="text-sm text-muted-foreground">Available Adapters</div>
          </div>
          <div>
            <div className="text-3xl font-bold gradient-text">1M+</div>
            <div className="text-sm text-muted-foreground">API Calls/Month</div>
          </div>
          <div>
            <div className="text-3xl font-bold gradient-text">4.9/5</div>
            <div className="text-sm text-muted-foreground">Average Rating</div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
