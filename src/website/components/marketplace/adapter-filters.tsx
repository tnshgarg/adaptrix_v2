'use client'

import { useState } from 'react'
import { Filter, X, Star, DollarSign, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { Badge } from '@/components/ui/badge'

interface AdapterFiltersProps {
  isOpen: boolean
  onClose: () => void
  filters: {
    priceRange: [number, number]
    minRating: number
    categories: string[]
    features: string[]
    showFreeOnly: boolean
    showPurchasedOnly: boolean
  }
  onFiltersChange: (filters: any) => void
}

export function AdapterFilters({ isOpen, onClose, filters, onFiltersChange }: AdapterFiltersProps) {
  const [localFilters, setLocalFilters] = useState(filters)

  const categories = [
    { id: 'code', name: 'Code Generation', count: 12 },
    { id: 'legal', name: 'Legal Analysis', count: 8 },
    { id: 'math', name: 'Mathematics', count: 6 },
    { id: 'general', name: 'General Purpose', count: 15 },
    { id: 'creative', name: 'Creative Writing', count: 9 },
    { id: 'data', name: 'Data Analysis', count: 7 }
  ]

  const features = [
    'Multi-language Support',
    'Real-time Processing',
    'Batch Operations',
    'Custom Training',
    'API Integration',
    'Cloud Deployment',
    'On-premise Support',
    'Advanced Analytics'
  ]

  const updateFilter = (key: string, value: any) => {
    const newFilters = { ...localFilters, [key]: value }
    setLocalFilters(newFilters)
    onFiltersChange(newFilters)
  }

  const toggleCategory = (categoryId: string) => {
    const newCategories = localFilters.categories.includes(categoryId)
      ? localFilters.categories.filter(id => id !== categoryId)
      : [...localFilters.categories, categoryId]
    updateFilter('categories', newCategories)
  }

  const toggleFeature = (feature: string) => {
    const newFeatures = localFilters.features.includes(feature)
      ? localFilters.features.filter(f => f !== feature)
      : [...localFilters.features, feature]
    updateFilter('features', newFeatures)
  }

  const clearAllFilters = () => {
    const defaultFilters = {
      priceRange: [0, 100] as [number, number],
      minRating: 0,
      categories: [],
      features: [],
      showFreeOnly: false,
      showPurchasedOnly: false
    }
    setLocalFilters(defaultFilters)
    onFiltersChange(defaultFilters)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 lg:relative lg:inset-auto">
      {/* Mobile overlay */}
      <div className="lg:hidden fixed inset-0 bg-black/50" onClick={onClose} />
      
      {/* Filter panel */}
      <Card className="lg:relative fixed right-0 top-0 h-full w-80 lg:w-full overflow-y-auto">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Filter className="w-5 h-5" />
            <span>Filters</span>
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm" onClick={clearAllFilters}>
              Clear All
            </Button>
            <Button variant="ghost" size="sm" onClick={onClose} className="lg:hidden">
              <X className="w-4 h-4" />
            </Button>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Price Range */}
          <div className="space-y-3">
            <Label className="flex items-center space-x-2">
              <DollarSign className="w-4 h-4" />
              <span>Price Range</span>
            </Label>
            <div className="px-2">
              <Slider
                value={localFilters.priceRange}
                onValueChange={(value) => updateFilter('priceRange', value as [number, number])}
                max={100}
                step={5}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-muted-foreground mt-1">
                <span>${localFilters.priceRange[0]}</span>
                <span>${localFilters.priceRange[1]}</span>
              </div>
            </div>
          </div>

          {/* Rating */}
          <div className="space-y-3">
            <Label className="flex items-center space-x-2">
              <Star className="w-4 h-4" />
              <span>Minimum Rating</span>
            </Label>
            <div className="px-2">
              <Slider
                value={[localFilters.minRating]}
                onValueChange={([value]) => updateFilter('minRating', value)}
                max={5}
                step={0.5}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-muted-foreground mt-1">
                <span>0 stars</span>
                <span>{localFilters.minRating} stars</span>
              </div>
            </div>
          </div>

          {/* Quick Filters */}
          <div className="space-y-3">
            <Label>Quick Filters</Label>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Free adapters only</span>
                <Switch
                  checked={localFilters.showFreeOnly}
                  onCheckedChange={(checked) => updateFilter('showFreeOnly', checked)}
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Purchased adapters only</span>
                <Switch
                  checked={localFilters.showPurchasedOnly}
                  onCheckedChange={(checked) => updateFilter('showPurchasedOnly', checked)}
                />
              </div>
            </div>
          </div>

          {/* Categories */}
          <div className="space-y-3">
            <Label>Categories</Label>
            <div className="space-y-2">
              {categories.map((category) => (
                <div
                  key={category.id}
                  className="flex items-center justify-between cursor-pointer p-2 rounded hover:bg-muted"
                  onClick={() => toggleCategory(category.id)}
                >
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={localFilters.categories.includes(category.id)}
                      onChange={() => toggleCategory(category.id)}
                      className="rounded"
                    />
                    <span className="text-sm">{category.name}</span>
                  </div>
                  <Badge variant="secondary" className="text-xs">
                    {category.count}
                  </Badge>
                </div>
              ))}
            </div>
          </div>

          {/* Features */}
          <div className="space-y-3">
            <Label>Features</Label>
            <div className="space-y-2">
              {features.map((feature) => (
                <div
                  key={feature}
                  className="flex items-center space-x-2 cursor-pointer p-2 rounded hover:bg-muted"
                  onClick={() => toggleFeature(feature)}
                >
                  <input
                    type="checkbox"
                    checked={localFilters.features.includes(feature)}
                    onChange={() => toggleFeature(feature)}
                    className="rounded"
                  />
                  <span className="text-sm">{feature}</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
