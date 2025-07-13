"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Bot,
  User,
  Settings,
  Plus,
  Trash2,
  Copy,
  ThumbsUp,
  ThumbsDown,
  MoreVertical,
  Sparkles,
  Zap,
  ChevronDown,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChatMessage, ChatSession, ChatSettings } from "@/lib/types";
import { ChatInput } from "@/components/chat/chat-input";
import { ChatMessage as ChatMessageComponent } from "@/components/chat/chat-message";
import { ChatSettings as ChatSettingsComponent } from "@/components/chat/chat-settings";
import { generateId } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function ChatPage() {
  // Single session state - no sidebar needed
  const [currentSession, setCurrentSession] = useState<ChatSession>({
    id: "1",
    title: "Code Generation Help",
    messages: [
      {
        id: "1",
        content:
          "Hello! I can help you with code generation, legal analysis, general questions, and mathematical problems. What would you like to work on today?",
        role: "assistant",
        timestamp: new Date(),
        metadata: {
          adapter: "general",
          model: "Qwen-3 1.7B",
          tokens: 45,
          processingTime: 1.2,
        },
      },
    ],
    createdAt: new Date(),
    updatedAt: new Date(),
    settings: {
      model: "Qwen-3 1.7B",
      adapters: ["general", "code", "legal"],
      temperature: 0.7,
      maxTokens: 150,
      topP: 0.9,
      topK: 50,
      useRag: true,
      ragTopK: 3,
    },
  });

  const [isLoading, setIsLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showAdapterSelector, setShowAdapterSelector] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Available adapters
  const availableAdapters = [
    {
      id: "general",
      name: "General",
      icon: "🤖",
      installed: true,
      description: "General purpose assistant",
    },
    {
      id: "code",
      name: "Code",
      icon: "💻",
      installed: true,
      description: "Code generation and debugging",
    },
    {
      id: "legal",
      name: "Legal",
      icon: "⚖️",
      installed: true,
      description: "Legal document analysis",
    },
    {
      id: "math",
      name: "Math Solver",
      icon: "🧮",
      installed: false,
      description: "Mathematical problem solving",
    },
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentSession?.messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: ChatMessage = {
      id: generateId(),
      content: content.trim(),
      role: "user",
      timestamp: new Date(),
    };

    // Add user message
    setCurrentSession((prev) => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      updatedAt: new Date(),
    }));

    setIsLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const assistantMessage: ChatMessage = {
        id: generateId(),
        content: generateMockResponse(content, currentSession.settings),
        role: "assistant",
        timestamp: new Date(),
        metadata: {
          adapter: selectAdapter(content),
          model: currentSession.settings.model,
          tokens: Math.floor(Math.random() * 100) + 50,
          processingTime: Math.random() * 3 + 1,
          ragSources: currentSession.settings.useRag
            ? [
                {
                  id: "1",
                  title: "Relevant Document",
                  content: "This is a relevant piece of information...",
                  score: 0.85,
                },
              ]
            : undefined,
        },
      };

      setCurrentSession((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
        updatedAt: new Date(),
      }));

      setIsLoading(false);
    }, 2000);
  };

  const selectAdapter = (content: string): string => {
    const lowerContent = content.toLowerCase();
    if (
      lowerContent.includes("code") ||
      lowerContent.includes("function") ||
      lowerContent.includes("python")
    ) {
      return "code";
    } else if (
      lowerContent.includes("legal") ||
      lowerContent.includes("contract") ||
      lowerContent.includes("law")
    ) {
      return "legal";
    } else if (
      lowerContent.includes("math") ||
      lowerContent.includes("calculate") ||
      lowerContent.includes("equation")
    ) {
      return "math";
    }
    return "general";
  };

  const generateMockResponse = (
    prompt: string,
    settings: ChatSettings
  ): string => {
    const adapter = selectAdapter(prompt);

    const responses = {
      code: "Here's a Python function that should help with your request:\n\n```python\ndef example_function(data):\n    # Process the data\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result\n```\n\nThis function filters positive numbers and doubles them. Would you like me to explain any part of it?",
      legal:
        "Based on legal analysis principles, here are the key considerations:\n\n1. **Contract Terms**: Review the specific clauses mentioned\n2. **Jurisdiction**: Consider applicable laws in your region\n3. **Precedent**: Look for similar cases or rulings\n\n⚠️ **Disclaimer**: This is for informational purposes only and does not constitute legal advice. Please consult with a qualified attorney for specific legal matters.",
      math: "Let me solve this step by step:\n\n**Given**: Your mathematical problem\n**Solution**:\n1. First, identify the variables\n2. Apply the appropriate formula\n3. Calculate the result\n\n**Answer**: The solution is X\n\nWould you like me to explain any of these steps in more detail?",
      general:
        "I understand your question. Based on the information available and the context you've provided, here's a comprehensive response that addresses your main points.\n\nKey insights:\n• Point 1: Relevant information\n• Point 2: Additional context\n• Point 3: Practical implications\n\nIs there anything specific you'd like me to elaborate on?",
    };

    return responses[adapter as keyof typeof responses] || responses.general;
  };

  const createNewSession = () => {
    const newSession: ChatSession = {
      id: generateId(),
      title: "New Chat",
      messages: [
        {
          id: generateId(),
          content: "Hello! How can I help you today?",
          role: "assistant",
          timestamp: new Date(),
          metadata: {
            adapter: "general",
            model: "Qwen-3 1.7B",
            tokens: 12,
            processingTime: 0.5,
          },
        },
      ],
      createdAt: new Date(),
      updatedAt: new Date(),
      settings: {
        model: "Qwen-3 1.7B",
        adapters: ["general"],
        temperature: 0.7,
        maxTokens: 150,
        topP: 0.9,
        topK: 50,
        useRag: true,
        ragTopK: 3,
      },
    };

    setCurrentSession(newSession);
  };

  const updateSessionSettings = (settings: ChatSettings) => {
    setCurrentSession((prev) => ({ ...prev, settings }));
  };

  const toggleAdapter = (adapterId: string) => {
    const updatedAdapters = currentSession.settings.adapters.includes(adapterId)
      ? currentSession.settings.adapters.filter((id) => id !== adapterId)
      : [...currentSession.settings.adapters, adapterId];

    updateSessionSettings({
      ...currentSession.settings,
      adapters: updatedAdapters,
    });
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Chat Header with Adapter Selector */}
      <div className="flex items-center justify-between p-6 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div>
          <h1 className="text-2xl font-bold">{currentSession.title}</h1>
          <p className="text-sm text-muted-foreground">
            {currentSession.messages.length} messages •{" "}
            {currentSession.settings.model}
          </p>
        </div>

        <div className="flex items-center space-x-3">
          {/* Adapter Composition */}
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-muted-foreground">
              Active Adapters:
            </span>
            <div className="flex items-center space-x-1">
              {currentSession.settings.adapters.map((adapterId) => {
                const adapter = availableAdapters.find(
                  (a) => a.id === adapterId
                );
                return adapter ? (
                  <Badge
                    key={adapterId}
                    variant="secondary"
                    className="flex items-center space-x-1"
                  >
                    <span>{adapter.icon}</span>
                    <span>{adapter.name}</span>
                  </Badge>
                ) : null;
              })}
            </div>

            <DropdownMenu
              open={showAdapterSelector}
              onOpenChange={setShowAdapterSelector}
            >
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <Plus className="w-4 h-4 mr-1" />
                  Manage
                  <ChevronDown className="w-4 h-4 ml-1" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-80">
                <div className="p-3">
                  <h4 className="font-medium mb-3">Available Adapters</h4>
                  <div className="space-y-2">
                    {availableAdapters.map((adapter) => (
                      <div
                        key={adapter.id}
                        className="flex items-center justify-between p-2 rounded-lg hover:bg-muted"
                      >
                        <div className="flex items-center space-x-3">
                          <span className="text-lg">{adapter.icon}</span>
                          <div>
                            <div className="font-medium text-sm">
                              {adapter.name}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {adapter.description}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          {adapter.installed ? (
                            <Button
                              variant={
                                currentSession.settings.adapters.includes(
                                  adapter.id
                                )
                                  ? "default"
                                  : "outline"
                              }
                              size="sm"
                              onClick={() => toggleAdapter(adapter.id)}
                            >
                              {currentSession.settings.adapters.includes(
                                adapter.id
                              )
                                ? "Active"
                                : "Add"}
                            </Button>
                          ) : (
                            <Badge variant="outline">Not Installed</Badge>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowSettings(!showSettings)}
          >
            <Settings className="w-4 h-4" />
          </Button>

          <Button variant="outline" size="sm" onClick={createNewSession}>
            <Plus className="w-4 h-4 mr-1" />
            New Chat
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto p-6 space-y-6">
          <AnimatePresence>
            {currentSession.messages.map((message) => (
              <ChatMessageComponent
                key={message.id}
                message={message}
                onCopy={() => navigator.clipboard.writeText(message.content)}
                onRegenerate={() => {
                  // Implement regeneration logic
                }}
              />
            ))}
          </AnimatePresence>

          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start space-x-4"
            >
              <div className="w-10 h-10 rounded-full bg-black flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <div className="bg-muted rounded-2xl p-4">
                  <div className="typing-indicator">
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Chat Input */}
      <div className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-4xl mx-auto p-6">
          <ChatInput
            onSendMessage={handleSendMessage}
            disabled={isLoading}
            placeholder="Type your message... (Adaptrix will automatically select the best adapter)"
          />
          <div className="flex items-center justify-between mt-3 text-xs text-muted-foreground">
            <div className="flex items-center space-x-4">
              <span>Press Enter to send, Shift+Enter for new line</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>AI Ready</span>
              </div>
            </div>
            <span>1 chat total</span>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <ChatSettingsComponent
          settings={currentSession.settings}
          onSettingsChange={updateSessionSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}
