'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Key, 
  Plus, 
  Eye, 
  EyeOff, 
  Copy, 
  Trash2, 
  Calendar,
  Shield,
  AlertTriangle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

export default function ApiKeysPage() {
  const [showKeys, setShowKeys] = useState<{[key: string]: boolean}>({})
  
  const apiKeys = [
    {
      id: '1',
      name: 'Production API Key',
      key: 'ak_prod_1234567890abcdef1234567890abcdef',
      created: '2024-01-15',
      lastUsed: '2024-01-20',
      permissions: ['read', 'write'],
      status: 'active',
      usage: 1250,
      limit: 10000
    },
    {
      id: '2',
      name: 'Development Key',
      key: 'ak_dev_abcdef1234567890abcdef1234567890',
      created: '2024-01-10',
      lastUsed: '2024-01-19',
      permissions: ['read'],
      status: 'active',
      usage: 450,
      limit: 5000
    },
    {
      id: '3',
      name: 'Testing Environment',
      key: 'ak_test_567890abcdef1234567890abcdef1234',
      created: '2024-01-05',
      lastUsed: '2024-01-18',
      permissions: ['read', 'write'],
      status: 'inactive',
      usage: 0,
      limit: 1000
    }
  ]

  const toggleKeyVisibility = (keyId: string) => {
    setShowKeys(prev => ({
      ...prev,
      [keyId]: !prev[keyId]
    }))
  }

  const maskKey = (key: string) => {
    return key.substring(0, 12) + '•'.repeat(20) + key.substring(key.length - 8)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    // You could add a toast notification here
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-600'
      case 'inactive': return 'bg-gray-600'
      case 'revoked': return 'bg-red-600'
      default: return 'bg-gray-600'
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">API Keys</h1>
          <p className="text-muted-foreground mt-2">
            Manage your API keys for programmatic access to Adaptrix
          </p>
        </div>
        <Button>
          <Plus className="w-4 h-4 mr-2" />
          Create New Key
        </Button>
      </div>

      {/* Security Notice */}
      <Card className="border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20">
        <CardContent className="p-4">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
            <div>
              <h3 className="font-medium text-yellow-800 dark:text-yellow-200">
                Keep your API keys secure
              </h3>
              <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
                Never share your API keys publicly or commit them to version control. 
                Store them securely and rotate them regularly.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Keys</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{apiKeys.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Active Keys</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {apiKeys.filter(k => k.status === 'active').length}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Usage</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {apiKeys.reduce((sum, key) => sum + key.usage, 0).toLocaleString()}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Usage Limit</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {apiKeys.reduce((sum, key) => sum + key.limit, 0).toLocaleString()}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Create New Key Form */}
      <Card>
        <CardHeader>
          <CardTitle>Create New API Key</CardTitle>
          <CardDescription>
            Generate a new API key with specific permissions and usage limits
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="keyName">Key Name</Label>
              <Input id="keyName" placeholder="e.g., Production API Key" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="permissions">Permissions</Label>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Select permissions" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="read">Read Only</SelectItem>
                  <SelectItem value="read-write">Read & Write</SelectItem>
                  <SelectItem value="admin">Admin Access</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="flex justify-end">
            <Button>Generate API Key</Button>
          </div>
        </CardContent>
      </Card>

      {/* API Keys List */}
      <div className="space-y-4">
        {apiKeys.map((apiKey, index) => (
          <motion.div
            key={apiKey.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <Card>
              <CardContent className="p-6">
                <div className="space-y-4">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-4">
                      <div className="w-12 h-12 bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 rounded-lg flex items-center justify-center">
                        <Key className="w-6 h-6 text-blue-600" />
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-lg font-semibold">{apiKey.name}</h3>
                          <Badge 
                            variant="secondary"
                            className={getStatusColor(apiKey.status)}
                          >
                            {apiKey.status}
                          </Badge>
                        </div>
                        
                        <div className="flex items-center space-x-6 text-sm text-muted-foreground mb-3">
                          <div className="flex items-center space-x-1">
                            <Calendar className="w-4 h-4" />
                            <span>Created {apiKey.created}</span>
                          </div>
                          <span>Last used {apiKey.lastUsed}</span>
                          <div className="flex items-center space-x-1">
                            <Shield className="w-4 h-4" />
                            <span>{apiKey.permissions.join(', ')}</span>
                          </div>
                        </div>

                        <div className="flex items-center space-x-2 mb-2">
                          <code className="bg-muted px-2 py-1 rounded text-sm font-mono">
                            {showKeys[apiKey.id] ? apiKey.key : maskKey(apiKey.key)}
                          </code>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleKeyVisibility(apiKey.id)}
                          >
                            {showKeys[apiKey.id] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => copyToClipboard(apiKey.key)}
                          >
                            <Copy className="w-4 h-4" />
                          </Button>
                        </div>

                        <div className="flex items-center space-x-4 text-sm">
                          <span>Usage: {apiKey.usage.toLocaleString()} / {apiKey.limit.toLocaleString()}</span>
                          <div className="flex-1 bg-muted rounded-full h-2 max-w-xs">
                            <div 
                              className="bg-blue-600 h-2 rounded-full" 
                              style={{ width: `${(apiKey.usage / apiKey.limit) * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Button variant="outline" size="sm">
                        Edit
                      </Button>
                      <Button variant="outline" size="sm">
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
