'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Puzzle, 
  Download, 
  Trash2, 
  Settings, 
  Star, 
  Calendar,
  Package,
  Search,
  Filter,
  MoreVertical
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import Link from 'next/link'

export default function AdaptersPage() {
  const [searchQuery, setSearchQuery] = useState('')
  
  const installedAdapters = [
    {
      id: '1',
      name: 'Code Generator Pro',
      description: 'Advanced code generation with debugging and optimization features.',
      version: '2.1.0',
      category: 'code',
      rating: 4.8,
      downloads: 15420,
      installedDate: '2024-01-15',
      status: 'active',
      size: '45 MB'
    },
    {
      id: '2',
      name: 'Legal Analyzer Pro',
      description: 'Professional legal document analysis with contract review.',
      version: '1.5.2',
      category: 'legal',
      rating: 4.9,
      downloads: 8750,
      installedDate: '2024-01-10',
      status: 'active',
      size: '62 MB'
    },
    {
      id: '3',
      name: 'General Assistant',
      description: 'Built-in general purpose AI assistant.',
      version: '1.0.0',
      category: 'general',
      rating: 4.7,
      downloads: 50000,
      installedDate: '2024-01-01',
      status: 'active',
      size: '32 MB'
    }
  ]

  const filteredAdapters = installedAdapters.filter(adapter =>
    adapter.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    adapter.description.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">My Adapters</h1>
          <p className="text-muted-foreground mt-2">
            Manage your installed AI adapters and their configurations
          </p>
        </div>
        <Link href="/marketplace">
          <Button>
            <Package className="w-4 h-4 mr-2" />
            Browse Marketplace
          </Button>
        </Link>
      </div>

      {/* Search and Filters */}
      <div className="flex items-center space-x-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search adapters..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        <Button variant="outline">
          <Filter className="w-4 h-4 mr-2" />
          Filter
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Adapters</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{installedAdapters.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Active</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {installedAdapters.filter(a => a.status === 'active').length}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Size</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">139 MB</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Avg Rating</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">4.8</div>
          </CardContent>
        </Card>
      </div>

      {/* Adapters List */}
      <div className="space-y-4">
        {filteredAdapters.map((adapter, index) => (
          <motion.div
            key={adapter.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <Card>
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    <div className="w-12 h-12 bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 rounded-lg flex items-center justify-center">
                      <Puzzle className="w-6 h-6 text-blue-600" />
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="text-lg font-semibold">{adapter.name}</h3>
                        <Badge variant="secondary">v{adapter.version}</Badge>
                        <Badge 
                          variant={adapter.status === 'active' ? 'default' : 'secondary'}
                          className={adapter.status === 'active' ? 'bg-green-600' : ''}
                        >
                          {adapter.status}
                        </Badge>
                      </div>
                      
                      <p className="text-muted-foreground mb-3">{adapter.description}</p>
                      
                      <div className="flex items-center space-x-6 text-sm text-muted-foreground">
                        <div className="flex items-center space-x-1">
                          <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                          <span>{adapter.rating}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Download className="w-4 h-4" />
                          <span>{adapter.downloads.toLocaleString()}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Calendar className="w-4 h-4" />
                          <span>Installed {adapter.installedDate}</span>
                        </div>
                        <span>{adapter.size}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">
                      <Settings className="w-4 h-4 mr-2" />
                      Configure
                    </Button>
                    <Button variant="outline" size="sm">
                      <MoreVertical className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {filteredAdapters.length === 0 && (
        <div className="text-center py-12">
          <Puzzle className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium mb-2">No adapters found</h3>
          <p className="text-muted-foreground mb-4">
            Try adjusting your search or browse the marketplace for new adapters.
          </p>
          <Link href="/marketplace">
            <Button>Browse Marketplace</Button>
          </Link>
        </div>
      )}
    </div>
  )
}
