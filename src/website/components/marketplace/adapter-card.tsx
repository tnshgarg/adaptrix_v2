'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Star, 
  Download, 
  ShoppingCart, 
  Check, 
  Eye,
  Code,
  Scale,
  Calculator,
  Brain,
  Sparkles
} from 'lucide-react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Adapter } from '@/lib/types'
import { formatPrice, cn } from '@/lib/utils'

interface AdapterCardProps {
  adapter: Adapter
  viewMode: 'grid' | 'list'
  isInCart: boolean
  onAddToCart: () => void
  onRemoveFromCart: () => void
}

export function AdapterCard({ 
  adapter, 
  viewMode, 
  isInCart, 
  onAddToCart, 
  onRemoveFromCart 
}: AdapterCardProps) {
  const [isHovered, setIsHovered] = useState(false)

  const getCategoryIcon = (category: string) => {
    const icons = {
      code: Code,
      legal: Scale,
      math: Calculator,
      general: Brain
    }
    return icons[category as keyof typeof icons] || Sparkles
  }

  const getCategoryColor = (category: string) => {
    const colors = {
      code: 'from-blue-500 to-cyan-500',
      legal: 'from-purple-500 to-pink-500',
      math: 'from-green-500 to-emerald-500',
      general: 'from-gray-500 to-slate-500'
    }
    return colors[category as keyof typeof colors] || colors.general
  }

  const CategoryIcon = getCategoryIcon(adapter.category)

  if (viewMode === 'list') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Card className="hover:shadow-lg transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center space-x-6">
              {/* Thumbnail */}
              <div className="w-24 h-24 rounded-lg bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 flex items-center justify-center shrink-0">
                <CategoryIcon className="w-8 h-8 text-blue-600" />
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold mb-1">{adapter.name}</h3>
                    <p className="text-muted-foreground text-sm mb-2 line-clamp-2">
                      {adapter.description}
                    </p>
                    
                    <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                      <div className="flex items-center space-x-1">
                        <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                        <span>{adapter.rating}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Download className="w-4 h-4" />
                        <span>{adapter.downloads.toLocaleString()}</span>
                      </div>
                      <span>by {adapter.author}</span>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4 ml-6">
                    <div className="text-right">
                      <div className="text-2xl font-bold">{formatPrice(adapter.price)}</div>
                      {adapter.isPurchased && (
                        <Badge variant="secondary" className="mt-1">
                          <Check className="w-3 h-3 mr-1" />
                          Purchased
                        </Badge>
                      )}
                    </div>

                    <div className="flex flex-col space-y-2">
                      <Link href={`/marketplace/${adapter.id}`}>
                        <Button variant="outline" size="sm">
                          <Eye className="w-4 h-4 mr-2" />
                          View
                        </Button>
                      </Link>
                      
                      {!adapter.isPurchased && (
                        <Button
                          size="sm"
                          onClick={isInCart ? onRemoveFromCart : onAddToCart}
                          variant={isInCart ? "secondary" : "default"}
                        >
                          <ShoppingCart className="w-4 h-4 mr-2" />
                          {isInCart ? 'Remove' : 'Add to Cart'}
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
    >
      <Card className="h-full hover:shadow-lg transition-all duration-300 group">
        {/* Thumbnail */}
        <div className="relative overflow-hidden rounded-t-lg">
          <div className={cn(
            "h-48 bg-gradient-to-r flex items-center justify-center",
            getCategoryColor(adapter.category)
          )}>
            <CategoryIcon className="w-16 h-16 text-white" />
          </div>
          
          {adapter.isPurchased && (
            <div className="absolute top-3 right-3">
              <Badge className="bg-green-600 text-white">
                <Check className="w-3 h-3 mr-1" />
                Purchased
              </Badge>
            </div>
          )}

          {/* Overlay on hover */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: isHovered ? 1 : 0 }}
            className="absolute inset-0 bg-black/20 flex items-center justify-center"
          >
            <Link href={`/marketplace/${adapter.id}`}>
              <Button variant="secondary">
                <Eye className="w-4 h-4 mr-2" />
                View Details
              </Button>
            </Link>
          </motion.div>
        </div>

        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <CardTitle className="text-lg line-clamp-1">{adapter.name}</CardTitle>
              <CardDescription className="line-clamp-2 mt-1">
                {adapter.description}
              </CardDescription>
            </div>
            <Badge variant="outline" className="ml-2 shrink-0">
              {adapter.category}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="pt-0">
          {/* Stats */}
          <div className="flex items-center justify-between text-sm text-muted-foreground mb-4">
            <div className="flex items-center space-x-1">
              <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
              <span>{adapter.rating}</span>
            </div>
            <div className="flex items-center space-x-1">
              <Download className="w-4 h-4" />
              <span>{adapter.downloads.toLocaleString()}</span>
            </div>
          </div>

          {/* Tags */}
          <div className="flex flex-wrap gap-1 mb-4">
            {adapter.tags.slice(0, 3).map((tag) => (
              <Badge key={tag} variant="secondary" className="text-xs">
                {tag}
              </Badge>
            ))}
            {adapter.tags.length > 3 && (
              <Badge variant="secondary" className="text-xs">
                +{adapter.tags.length - 3}
              </Badge>
            )}
          </div>

          {/* Price and Actions */}
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold">{formatPrice(adapter.price)}</div>
            
            {!adapter.isPurchased && (
              <Button
                size="sm"
                onClick={isInCart ? onRemoveFromCart : onAddToCart}
                variant={isInCart ? "secondary" : "default"}
                className="shrink-0"
              >
                <ShoppingCart className="w-4 h-4 mr-2" />
                {isInCart ? 'Remove' : 'Add'}
              </Button>
            )}
          </div>

          {/* Author */}
          <div className="text-xs text-muted-foreground mt-2">
            by {adapter.author}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
