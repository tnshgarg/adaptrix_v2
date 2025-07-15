'use client'

import { Bell, Search } from 'lucide-react'
import { UserButton } from '@clerk/nextjs'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export function DashboardHeader() {
  return (
    <header className="bg-background border-b px-4 sm:px-6 lg:px-8">
      <div className="flex h-16 items-center justify-between">
        {/* Search */}
        <div className="flex-1 max-w-md">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search adapters, documents..."
              className="pl-10"
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-4">
          <Button variant="ghost" size="sm">
            <Bell className="w-4 h-4" />
          </Button>
          <UserButton afterSignOutUrl="/" />
        </div>
      </div>
    </header>
  )
}
