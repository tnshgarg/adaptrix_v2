'use client'

import { useState } from 'react'
import { X, Settings, Sliders, Database, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import { ChatSettings as ChatSettingsType } from '@/lib/types'

interface ChatSettingsProps {
  settings: ChatSettingsType
  onSettingsChange: (settings: ChatSettingsType) => void
  onClose: () => void
}

export function ChatSettings({ settings, onSettingsChange, onClose }: ChatSettingsProps) {
  const [localSettings, setLocalSettings] = useState<ChatSettingsType>(settings)

  const updateSetting = <K extends keyof ChatSettingsType>(
    key: K,
    value: ChatSettingsType[K]
  ) => {
    const newSettings = { ...localSettings, [key]: value }
    setLocalSettings(newSettings)
    onSettingsChange(newSettings)
  }

  return (
    <div className="w-80 bg-background border-l flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-2">
          <Settings className="w-5 h-5" />
          <h3 className="font-semibold">Chat Settings</h3>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="w-4 h-4" />
        </Button>
      </div>

      {/* Settings Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Model Settings */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center space-x-2">
              <Zap className="w-4 h-4" />
              <span>Model Configuration</span>
            </CardTitle>
            <CardDescription>
              Adjust how the AI generates responses
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="temperature">Temperature</Label>
                <span className="text-sm text-muted-foreground">
                  {localSettings.temperature}
                </span>
              </div>
              <Slider
                id="temperature"
                min={0}
                max={2}
                step={0.1}
                value={[localSettings.temperature]}
                onValueChange={([value]) => updateSetting('temperature', value)}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Higher values make output more random, lower values more focused
              </p>
            </div>

            {/* Max Tokens */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="maxTokens">Max Tokens</Label>
                <span className="text-sm text-muted-foreground">
                  {localSettings.maxTokens}
                </span>
              </div>
              <Slider
                id="maxTokens"
                min={50}
                max={2048}
                step={50}
                value={[localSettings.maxTokens]}
                onValueChange={([value]) => updateSetting('maxTokens', value)}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Maximum number of tokens to generate
              </p>
            </div>

            {/* Top P */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="topP">Top P</Label>
                <span className="text-sm text-muted-foreground">
                  {localSettings.topP}
                </span>
              </div>
              <Slider
                id="topP"
                min={0}
                max={1}
                step={0.05}
                value={[localSettings.topP]}
                onValueChange={([value]) => updateSetting('topP', value)}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Nucleus sampling parameter
              </p>
            </div>

            {/* Top K */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="topK">Top K</Label>
                <span className="text-sm text-muted-foreground">
                  {localSettings.topK}
                </span>
              </div>
              <Slider
                id="topK"
                min={1}
                max={100}
                step={1}
                value={[localSettings.topK]}
                onValueChange={([value]) => updateSetting('topK', value)}
                className="w-full"
              />
              <p className="text-xs text-muted-foreground">
                Limits vocabulary to top K tokens
              </p>
            </div>
          </CardContent>
        </Card>

        {/* RAG Settings */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center space-x-2">
              <Database className="w-4 h-4" />
              <span>RAG Configuration</span>
            </CardTitle>
            <CardDescription>
              Control document retrieval and context
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Enable RAG */}
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="useRag">Enable RAG</Label>
                <p className="text-xs text-muted-foreground">
                  Use document retrieval for enhanced responses
                </p>
              </div>
              <Switch
                id="useRag"
                checked={localSettings.useRag}
                onCheckedChange={(checked) => updateSetting('useRag', checked)}
              />
            </div>

            {/* RAG Top K */}
            {localSettings.useRag && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="ragTopK">Documents to Retrieve</Label>
                  <span className="text-sm text-muted-foreground">
                    {localSettings.ragTopK}
                  </span>
                </div>
                <Slider
                  id="ragTopK"
                  min={1}
                  max={10}
                  step={1}
                  value={[localSettings.ragTopK]}
                  onValueChange={([value]) => updateSetting('ragTopK', value)}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  Number of relevant documents to include
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Model Info */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Model Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Model:</span>
              <span>{localSettings.model}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Active Adapters:</span>
              <span>{localSettings.adapters.length}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Context Length:</span>
              <span>32K tokens</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Footer */}
      <div className="p-4 border-t">
        <Button
          variant="outline"
          size="sm"
          className="w-full"
          onClick={() => {
            // Reset to defaults
            const defaultSettings: ChatSettingsType = {
              model: 'Qwen-3 1.7B',
              adapters: ['general'],
              temperature: 0.7,
              maxTokens: 150,
              topP: 0.9,
              topK: 50,
              useRag: true,
              ragTopK: 3
            }
            setLocalSettings(defaultSettings)
            onSettingsChange(defaultSettings)
          }}
        >
          Reset to Defaults
        </Button>
      </div>
    </div>
  )
}
