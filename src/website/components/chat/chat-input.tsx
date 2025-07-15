'use client'

import { useState, useRef, KeyboardEvent } from 'react'
import { Send, Paperclip, Mic, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
  placeholder?: string
  maxLength?: number
}

export function ChatInput({ 
  onSendMessage, 
  disabled = false, 
  placeholder = "Type your message...",
  maxLength = 4000
}: ChatInputProps) {
  const [message, setMessage] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = () => {
    if (message.trim() && !disabled) {
      onSendMessage(message.trim())
      setMessage('')
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value
    if (value.length <= maxLength) {
      setMessage(value)
      
      // Auto-resize textarea
      const textarea = e.target
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }

  const toggleRecording = () => {
    setIsRecording(!isRecording)
    // Implement voice recording logic here
  }

  return (
    <div className="relative">
      {/* Input Container */}
      <div className="flex items-end space-x-2 p-3 bg-background border rounded-lg shadow-sm">
        {/* Attachment Button */}
        <Button
          variant="ghost"
          size="sm"
          className="shrink-0"
          disabled={disabled}
        >
          <Paperclip className="w-4 h-4" />
        </Button>

        {/* Text Input */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            className={cn(
              "w-full resize-none border-0 bg-transparent p-0 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-0",
              "min-h-[20px] max-h-[200px]"
            )}
            rows={1}
          />
          
          {/* Character Count */}
          {message.length > maxLength * 0.8 && (
            <div className="absolute -top-6 right-0 text-xs text-muted-foreground">
              {message.length}/{maxLength}
            </div>
          )}
        </div>

        {/* Voice Recording Button */}
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            "shrink-0",
            isRecording && "text-red-500 animate-pulse"
          )}
          onClick={toggleRecording}
          disabled={disabled}
        >
          {isRecording ? (
            <Square className="w-4 h-4" />
          ) : (
            <Mic className="w-4 h-4" />
          )}
        </Button>

        {/* Send Button */}
        <Button
          size="sm"
          onClick={handleSubmit}
          disabled={disabled || !message.trim()}
          className="shrink-0"
        >
          <Send className="w-4 h-4" />
        </Button>
      </div>

      {/* Quick Actions */}
      <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
        <div className="flex items-center space-x-4">
          <span>Press Enter to send, Shift+Enter for new line</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span>AI Ready</span>
        </div>
      </div>
    </div>
  )
}
