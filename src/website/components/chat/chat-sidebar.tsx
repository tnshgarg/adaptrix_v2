'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Plus, MessageSquare, Trash2, MoreVertical, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ChatSession } from '@/lib/types'
import { formatRelativeTime, cn } from '@/lib/utils'

interface ChatSidebarProps {
  sessions: ChatSession[]
  currentSessionId: string
  onSessionSelect: (sessionId: string) => void
  onNewSession: () => void
  onDeleteSession: (sessionId: string) => void
}

export function ChatSidebar({
  sessions,
  currentSessionId,
  onSessionSelect,
  onNewSession,
  onDeleteSession
}: ChatSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')

  const filteredSessions = sessions.filter(session =>
    session.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    session.messages.some(msg => 
      msg.content.toLowerCase().includes(searchQuery.toLowerCase())
    )
  )

  return (
    <div className="w-80 bg-muted/30 border-r flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b">
        <Button
          onClick={onNewSession}
          className="w-full justify-start"
          variant="default"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Chat
        </Button>
      </div>

      {/* Search */}
      <div className="p-4 border-b">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search chats..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-2 space-y-1">
          {filteredSessions.map((session) => (
            <motion.div
              key={session.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
              className={cn(
                "group relative p-3 rounded-lg cursor-pointer transition-colors",
                session.id === currentSessionId
                  ? "bg-primary text-primary-foreground"
                  : "hover:bg-muted"
              )}
              onClick={() => onSessionSelect(session.id)}
            >
              <div className="flex items-start space-x-3">
                <MessageSquare className="w-4 h-4 mt-1 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm truncate">
                    {session.title}
                  </div>
                  <div className={cn(
                    "text-xs mt-1 truncate",
                    session.id === currentSessionId
                      ? "text-primary-foreground/70"
                      : "text-muted-foreground"
                  )}>
                    {session.messages[session.messages.length - 1]?.content || 'No messages'}
                  </div>
                  <div className={cn(
                    "text-xs mt-1",
                    session.id === currentSessionId
                      ? "text-primary-foreground/50"
                      : "text-muted-foreground"
                  )}>
                    {formatRelativeTime(session.updatedAt)}
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteSession(session.id)
                  }}
                  className={cn(
                    "h-6 w-6 p-0",
                    session.id === currentSessionId
                      ? "hover:bg-primary-foreground/20"
                      : "hover:bg-muted"
                  )}
                >
                  <Trash2 className="w-3 h-3" />
                </Button>
              </div>
            </motion.div>
          ))}

          {filteredSessions.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No chats found</p>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t">
        <div className="text-xs text-muted-foreground text-center">
          {sessions.length} chat{sessions.length !== 1 ? 's' : ''} total
        </div>
      </div>
    </div>
  )
}
