'use client'

import { useState } from 'react'
import { Check, ChevronDown, Code, Scale, Calculator, Brain, Puzzle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

interface AdapterSelectorProps {
  selectedAdapters: string[]
  onAdaptersChange: (adapters: string[]) => void
}

export function AdapterSelector({ selectedAdapters, onAdaptersChange }: AdapterSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)

  const availableAdapters = [
    {
      id: 'general',
      name: 'General Assistant',
      description: 'General purpose AI assistant',
      icon: Brain,
      color: 'from-gray-500 to-slate-500',
      installed: true
    },
    {
      id: 'code',
      name: 'Code Generator',
      description: 'Advanced code generation and debugging',
      icon: Code,
      color: 'from-blue-500 to-cyan-500',
      installed: true
    },
    {
      id: 'legal',
      name: 'Legal Analyzer',
      description: 'Legal document analysis and review',
      icon: Scale,
      color: 'from-purple-500 to-pink-500',
      installed: true
    },
    {
      id: 'math',
      name: 'Math Solver',
      description: 'Mathematical reasoning and calculations',
      icon: Calculator,
      color: 'from-green-500 to-emerald-500',
      installed: false
    }
  ]

  const installedAdapters = availableAdapters.filter(adapter => adapter.installed)

  const toggleAdapter = (adapterId: string) => {
    if (selectedAdapters.includes(adapterId)) {
      onAdaptersChange(selectedAdapters.filter(id => id !== adapterId))
    } else {
      onAdaptersChange([...selectedAdapters, adapterId])
    }
  }

  const getSelectedAdapterNames = () => {
    return selectedAdapters
      .map(id => availableAdapters.find(a => a.id === id)?.name)
      .filter(Boolean)
      .join(', ')
  }

  return (
    <div className="relative">
      <Button
        variant="outline"
        onClick={() => setIsOpen(!isOpen)}
        className="justify-between min-w-[200px]"
      >
        <div className="flex items-center space-x-2">
          <Puzzle className="w-4 h-4" />
          <span className="truncate">
            {selectedAdapters.length === 0 
              ? 'Select Adapters' 
              : selectedAdapters.length === 1
                ? getSelectedAdapterNames()
                : `${selectedAdapters.length} adapters`
            }
          </span>
        </div>
        <ChevronDown className={cn(
          "w-4 h-4 transition-transform",
          isOpen && "rotate-180"
        )} />
      </Button>

      {isOpen && (
        <Card className="absolute top-full left-0 right-0 mt-2 z-50 shadow-lg">
          <CardContent className="p-4 space-y-3">
            <div className="text-sm font-medium mb-3">Available Adapters</div>
            
            {installedAdapters.map((adapter) => (
              <div
                key={adapter.id}
                className={cn(
                  "flex items-center space-x-3 p-3 rounded-lg cursor-pointer transition-colors",
                  selectedAdapters.includes(adapter.id)
                    ? "bg-primary/10 border border-primary/20"
                    : "hover:bg-muted"
                )}
                onClick={() => toggleAdapter(adapter.id)}
              >
                <div className={`w-8 h-8 rounded-lg bg-gradient-to-r ${adapter.color} flex items-center justify-center`}>
                  <adapter.icon className="w-4 h-4 text-white" />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm">{adapter.name}</div>
                  <div className="text-xs text-muted-foreground truncate">
                    {adapter.description}
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Badge variant="secondary" className="text-xs">
                    Installed
                  </Badge>
                  
                  {selectedAdapters.includes(adapter.id) && (
                    <div className="w-4 h-4 rounded-full bg-primary flex items-center justify-center">
                      <Check className="w-3 h-3 text-primary-foreground" />
                    </div>
                  )}
                </div>
              </div>
            ))}

            {availableAdapters.filter(a => !a.installed).length > 0 && (
              <>
                <div className="border-t pt-3">
                  <div className="text-sm font-medium mb-3 text-muted-foreground">
                    Available in Marketplace
                  </div>
                  
                  {availableAdapters.filter(a => !a.installed).map((adapter) => (
                    <div
                      key={adapter.id}
                      className="flex items-center space-x-3 p-3 rounded-lg opacity-60"
                    >
                      <div className={`w-8 h-8 rounded-lg bg-gradient-to-r ${adapter.color} flex items-center justify-center`}>
                        <adapter.icon className="w-4 h-4 text-white" />
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">{adapter.name}</div>
                        <div className="text-xs text-muted-foreground truncate">
                          {adapter.description}
                        </div>
                      </div>

                      <Badge variant="outline" className="text-xs">
                        Not Installed
                      </Badge>
                    </div>
                  ))}
                </div>
              </>
            )}

            <div className="border-t pt-3">
              <Button
                variant="outline"
                size="sm"
                className="w-full"
                onClick={() => setIsOpen(false)}
              >
                Done
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
