'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Search, 
  Filter, 
  Star, 
  Download, 
  ShoppingCart,
  Grid,
  List,
  SlidersHorizontal,
  Code,
  Scale,
  Calculator,
  Brain,
  Sparkles
} from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Adapter } from '@/lib/types'
import { formatPrice } from '@/lib/utils'
import { Navbar } from '@/components/layout/navbar'
import { AdapterCard } from '@/components/marketplace/adapter-card'
import { AdapterFilters } from '@/components/marketplace/adapter-filters'
import { CartSidebar } from '@/components/marketplace/cart-sidebar'

// Mock data for adapters
const mockAdapters: Adapter[] = [
  {
    id: '1',
    name: 'Advanced Code Generator',
    description: 'State-of-the-art code generation for Python, JavaScript, TypeScript, and more. Includes debugging and optimization features.',
    category: 'code',
    price: 29.99,
    rating: 4.8,
    downloads: 15420,
    author: 'CodeCraft AI',
    version: '2.1.0',
    size: '45 MB',
    tags: ['python', 'javascript', 'typescript', 'debugging', 'optimization'],
    features: [
      'Multi-language support',
      'Code optimization',
      'Bug detection',
      'Documentation generation',
      'Test case creation'
    ],
    compatibility: ['Qwen-3 1.7B', 'Qwen-2 7B'],
    createdAt: new Date('2024-01-15'),
    updatedAt: new Date('2024-01-20'),
    isPurchased: false,
    isInstalled: false,
    thumbnail: 'https://via.placeholder.com/300x200?text=Code+Generator'
  },
  {
    id: '2',
    name: 'Legal Document Analyzer Pro',
    description: 'Professional-grade legal document analysis with contract review, compliance checking, and risk assessment.',
    category: 'legal',
    price: 49.99,
    rating: 4.9,
    downloads: 8750,
    author: 'LegalTech Solutions',
    version: '1.5.2',
    size: '62 MB',
    tags: ['contracts', 'compliance', 'risk-assessment', 'legal-research'],
    features: [
      'Contract analysis',
      'Compliance checking',
      'Risk assessment',
      'Legal research',
      'Citation verification'
    ],
    compatibility: ['Qwen-3 1.7B', 'Qwen-2 7B'],
    createdAt: new Date('2024-01-10'),
    updatedAt: new Date('2024-01-18'),
    isPurchased: true,
    isInstalled: true,
    thumbnail: 'https://via.placeholder.com/300x200?text=Legal+Analyzer'
  },
  {
    id: '3',
    name: 'Mathematical Reasoning Engine',
    description: 'Advanced mathematical problem solving with step-by-step explanations, graph plotting, and formula derivation.',
    category: 'math',
    price: 19.99,
    rating: 4.7,
    downloads: 12300,
    author: 'MathAI Labs',
    version: '3.0.1',
    size: '38 MB',
    tags: ['calculus', 'algebra', 'statistics', 'geometry', 'plotting'],
    features: [
      'Step-by-step solutions',
      'Graph plotting',
      'Formula derivation',
      'Statistical analysis',
      'Equation solving'
    ],
    compatibility: ['Qwen-3 1.7B', 'Qwen-2 7B'],
    createdAt: new Date('2024-01-12'),
    updatedAt: new Date('2024-01-22'),
    isPurchased: false,
    isInstalled: false,
    thumbnail: 'https://via.placeholder.com/300x200?text=Math+Engine'
  },
  {
    id: '4',
    name: 'Creative Writing Assistant',
    description: 'Enhance your creative writing with style suggestions, plot development, character creation, and narrative flow optimization.',
    category: 'general',
    price: 24.99,
    rating: 4.6,
    downloads: 9850,
    author: 'CreativeAI Studio',
    version: '1.8.0',
    size: '41 MB',
    tags: ['creative-writing', 'storytelling', 'character-development', 'plot'],
    features: [
      'Style enhancement',
      'Plot development',
      'Character creation',
      'Narrative flow',
      'Genre adaptation'
    ],
    compatibility: ['Qwen-3 1.7B', 'Qwen-2 7B'],
    createdAt: new Date('2024-01-08'),
    updatedAt: new Date('2024-01-19'),
    isPurchased: false,
    isInstalled: false,
    thumbnail: 'https://via.placeholder.com/300x200?text=Creative+Writing'
  }
]

export default function MarketplacePage() {
  const [adapters, setAdapters] = useState<Adapter[]>(mockAdapters)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [sortBy, setSortBy] = useState<string>('popular')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [showFilters, setShowFilters] = useState(false)
  const [showCart, setShowCart] = useState(false)
  const [cartItems, setCartItems] = useState<string[]>([])

  const categories = [
    { id: 'all', name: 'All Adapters', icon: Sparkles },
    { id: 'code', name: 'Code Generation', icon: Code },
    { id: 'legal', name: 'Legal Analysis', icon: Scale },
    { id: 'math', name: 'Mathematics', icon: Calculator },
    { id: 'general', name: 'General Purpose', icon: Brain }
  ]

  const filteredAdapters = adapters.filter(adapter => {
    const matchesSearch = adapter.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         adapter.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         adapter.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    
    const matchesCategory = selectedCategory === 'all' || adapter.category === selectedCategory
    
    return matchesSearch && matchesCategory
  })

  const sortedAdapters = [...filteredAdapters].sort((a, b) => {
    switch (sortBy) {
      case 'popular':
        return b.downloads - a.downloads
      case 'rating':
        return b.rating - a.rating
      case 'price-low':
        return a.price - b.price
      case 'price-high':
        return b.price - a.price
      case 'newest':
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      default:
        return 0
    }
  })

  const addToCart = (adapterId: string) => {
    setCartItems(prev => [...prev, adapterId])
  }

  const removeFromCart = (adapterId: string) => {
    setCartItems(prev => prev.filter(id => id !== adapterId))
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      
      <div className="pt-16">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-gray-900 dark:to-purple-900 py-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center"
            >
              <h1 className="text-4xl font-bold mb-4">
                Adapter <span className="gradient-text">Marketplace</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Discover and purchase specialized AI adapters to enhance your Adaptrix system
              </p>
            </motion.div>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col lg:flex-row gap-6">
            {/* Search Bar */}
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  placeholder="Search adapters..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 h-12"
                />
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center space-x-4">
              <Button
                variant="outline"
                onClick={() => setShowFilters(!showFilters)}
                className="h-12"
              >
                <SlidersHorizontal className="w-4 h-4 mr-2" />
                Filters
              </Button>

              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="h-12 px-3 rounded-md border border-input bg-background"
              >
                <option value="popular">Most Popular</option>
                <option value="rating">Highest Rated</option>
                <option value="price-low">Price: Low to High</option>
                <option value="price-high">Price: High to Low</option>
                <option value="newest">Newest</option>
              </select>

              <div className="flex border rounded-md">
                <Button
                  variant={viewMode === 'grid' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('grid')}
                >
                  <Grid className="w-4 h-4" />
                </Button>
                <Button
                  variant={viewMode === 'list' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setViewMode('list')}
                >
                  <List className="w-4 h-4" />
                </Button>
              </div>

              <Button
                variant="outline"
                onClick={() => setShowCart(true)}
                className="h-12"
              >
                <ShoppingCart className="w-4 h-4 mr-2" />
                Cart ({cartItems.length})
              </Button>
            </div>
          </div>

          {/* Category Tabs */}
          <div className="flex space-x-1 mt-6 bg-muted p-1 rounded-lg w-fit">
            {categories.map((category) => (
              <Button
                key={category.id}
                variant={selectedCategory === category.id ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setSelectedCategory(category.id)}
                className="flex items-center space-x-2"
              >
                <category.icon className="w-4 h-4" />
                <span>{category.name}</span>
              </Button>
            ))}
          </div>
        </div>

        {/* Adapters Grid */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
          <div className={`grid gap-6 ${
            viewMode === 'grid' 
              ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3' 
              : 'grid-cols-1'
          }`}>
            {sortedAdapters.map((adapter) => (
              <AdapterCard
                key={adapter.id}
                adapter={adapter}
                viewMode={viewMode}
                isInCart={cartItems.includes(adapter.id)}
                onAddToCart={() => addToCart(adapter.id)}
                onRemoveFromCart={() => removeFromCart(adapter.id)}
              />
            ))}
          </div>

          {sortedAdapters.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">No adapters found matching your criteria.</p>
            </div>
          )}
        </div>
      </div>

      {/* Cart Sidebar */}
      <CartSidebar
        isOpen={showCart}
        onClose={() => setShowCart(false)}
        cartItems={cartItems.map(id => mockAdapters.find(a => a.id === id)!).filter(Boolean)}
        onRemoveItem={removeFromCart}
      />
    </div>
  )
}
