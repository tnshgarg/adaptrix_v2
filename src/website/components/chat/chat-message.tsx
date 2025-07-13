'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Bot, 
  User, 
  Copy, 
  ThumbsUp, 
  ThumbsDown, 
  MoreVertical,
  Clock,
  Zap,
  FileText,
  Check
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { ChatMessage as ChatMessageType } from '@/lib/types'
import { formatRelativeTime, cn } from '@/lib/utils'

interface ChatMessageProps {
  message: ChatMessageType
  onCopy: () => void
  onRegenerate: () => void
  onFeedback?: (type: 'positive' | 'negative') => void
}

export function ChatMessage({ 
  message, 
  onCopy, 
  onRegenerate, 
  onFeedback 
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState<'positive' | 'negative' | null>(null)

  const isAssistant = message.role === 'assistant'
  const isUser = message.role === 'user'

  const handleCopy = async () => {
    await onCopy()
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleFeedback = (type: 'positive' | 'negative') => {
    setFeedback(type)
    onFeedback?.(type)
  }

  const getAdapterColor = (adapter?: string) => {
    const colors = {
      code: 'from-blue-500 to-cyan-500',
      legal: 'from-purple-500 to-pink-500',
      math: 'from-green-500 to-emerald-500',
      general: 'from-gray-500 to-slate-500'
    }
    return colors[adapter as keyof typeof colors] || colors.general
  }

  const getAdapterName = (adapter?: string) => {
    const names = {
      code: 'Code Generator',
      legal: 'Legal Analyzer',
      math: 'Math Solver',
      general: 'General Assistant'
    }
    return names[adapter as keyof typeof names] || 'AI Assistant'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex items-start space-x-3 group",
        isUser && "flex-row-reverse space-x-reverse"
      )}
    >
      {/* Avatar */}
      <div className={cn(
        "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
        isAssistant && `bg-gradient-to-r ${getAdapterColor(message.metadata?.adapter)}`,
        isUser && "bg-muted"
      )}>
        {isAssistant ? (
          <Bot className="w-4 h-4 text-white" />
        ) : (
          <User className="w-4 h-4" />
        )}
      </div>

      {/* Message Content */}
      <div className={cn(
        "flex-1 max-w-3xl",
        isUser && "flex flex-col items-end"
      )}>
        {/* Message Header */}
        {isAssistant && message.metadata && (
          <div className="flex items-center space-x-2 mb-2 text-xs text-muted-foreground">
            <span className="font-medium">
              {getAdapterName(message.metadata.adapter)}
            </span>
            <span>•</span>
            <span>{message.metadata.model}</span>
            {message.metadata.processingTime && (
              <>
                <span>•</span>
                <div className="flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>{message.metadata.processingTime.toFixed(1)}s</span>
                </div>
              </>
            )}
            {message.metadata.tokens && (
              <>
                <span>•</span>
                <div className="flex items-center space-x-1">
                  <Zap className="w-3 h-3" />
                  <span>{message.metadata.tokens} tokens</span>
                </div>
              </>
            )}
          </div>
        )}

        {/* Message Bubble */}
        <Card className={cn(
          "p-4 relative",
          isAssistant && "bg-muted/50",
          isUser && "bg-primary text-primary-foreground ml-12"
        )}>
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <div className="whitespace-pre-wrap break-words">
              {message.content}
            </div>
          </div>

          {/* RAG Sources */}
          {isAssistant && message.metadata?.ragSources && message.metadata.ragSources.length > 0 && (
            <div className="mt-4 pt-4 border-t">
              <div className="flex items-center space-x-2 mb-2">
                <FileText className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium text-muted-foreground">
                  Sources
                </span>
              </div>
              <div className="space-y-2">
                {message.metadata.ragSources.map((source, index) => (
                  <div key={source.id} className="text-xs bg-background rounded p-2">
                    <div className="font-medium">{source.title}</div>
                    <div className="text-muted-foreground truncate">
                      {source.content}
                    </div>
                    <div className="text-muted-foreground mt-1">
                      Relevance: {(source.score * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* Message Actions */}
        <div className={cn(
          "flex items-center space-x-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity",
          isUser && "flex-row-reverse"
        )}>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="h-8 px-2"
          >
            {copied ? (
              <Check className="w-3 h-3" />
            ) : (
              <Copy className="w-3 h-3" />
            )}
          </Button>

          {isAssistant && (
            <>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleFeedback('positive')}
                className={cn(
                  "h-8 px-2",
                  feedback === 'positive' && "text-green-600"
                )}
              >
                <ThumbsUp className="w-3 h-3" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleFeedback('negative')}
                className={cn(
                  "h-8 px-2",
                  feedback === 'negative' && "text-red-600"
                )}
              >
                <ThumbsDown className="w-3 h-3" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={onRegenerate}
                className="h-8 px-2"
              >
                <MoreVertical className="w-3 h-3" />
              </Button>
            </>
          )}
        </div>

        {/* Timestamp */}
        <div className={cn(
          "text-xs text-muted-foreground mt-1",
          isUser && "text-right"
        )}>
          {formatRelativeTime(message.timestamp)}
        </div>
      </div>
    </motion.div>
  )
}
