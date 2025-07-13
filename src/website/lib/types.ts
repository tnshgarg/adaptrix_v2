export interface User {
  id: string
  email: string
  name: string
  avatar?: string
  createdAt: Date
  subscription?: Subscription
}

export interface Subscription {
  id: string
  plan: 'free' | 'pro' | 'enterprise'
  status: 'active' | 'canceled' | 'past_due'
  currentPeriodEnd: Date
  cancelAtPeriodEnd: boolean
}

export interface Adapter {
  id: string
  name: string
  description: string
  category: 'code' | 'legal' | 'general' | 'math' | 'custom'
  price: number
  rating: number
  downloads: number
  author: string
  version: string
  size: string
  tags: string[]
  features: string[]
  compatibility: string[]
  createdAt: Date
  updatedAt: Date
  isPurchased?: boolean
  isInstalled?: boolean
  thumbnail?: string
  screenshots?: string[]
  documentation?: string
  changelog?: string
}

export interface CartItem {
  adapter: Adapter
  quantity: number
}

export interface Cart {
  items: CartItem[]
  total: number
  subtotal: number
  tax: number
}

export interface ChatMessage {
  id: string
  content: string
  role: 'user' | 'assistant' | 'system'
  timestamp: Date
  metadata?: {
    adapter?: string
    model?: string
    tokens?: number
    processingTime?: number
    ragSources?: RagSource[]
  }
}

export interface RagSource {
  id: string
  title: string
  content: string
  score: number
  metadata?: Record<string, any>
}

export interface ChatSession {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: Date
  updatedAt: Date
  settings: ChatSettings
}

export interface ChatSettings {
  model: string
  adapters: string[]
  temperature: number
  maxTokens: number
  topP: number
  topK: number
  useRag: boolean
  ragTopK: number
}

export interface Document {
  id: string
  name: string
  content: string
  type: 'text' | 'pdf' | 'docx' | 'md'
  size: number
  uploadedAt: Date
  processedAt?: Date
  status: 'uploading' | 'processing' | 'ready' | 'error'
  chunks?: number
  metadata?: Record<string, any>
}

export interface ApiKey {
  id: string
  name: string
  key: string
  permissions: string[]
  createdAt: Date
  lastUsed?: Date
  isActive: boolean
  usage: {
    requests: number
    tokens: number
    resetDate: Date
  }
}

export interface UserAdapter {
  id: string
  adapterId: string
  adapter: Adapter
  purchasedAt: Date
  isInstalled: boolean
  installedAt?: Date
  settings?: Record<string, any>
}

export interface GenerationRequest {
  prompt: string
  maxTokens?: number
  temperature?: number
  topP?: number
  topK?: number
  adapters?: string[]
  useRag?: boolean
  ragTopK?: number
  stopSequences?: string[]
}

export interface GenerationResponse {
  id: string
  content: string
  metadata: {
    model: string
    adapters: string[]
    tokens: number
    processingTime: number
    ragSources?: RagSource[]
  }
  createdAt: Date
}

export interface SystemStats {
  totalUsers: number
  totalAdapters: number
  totalGenerations: number
  averageResponseTime: number
  uptime: number
  version: string
}

export interface NotificationSettings {
  email: boolean
  push: boolean
  marketing: boolean
  updates: boolean
  security: boolean
}

export interface UserSettings {
  theme: 'light' | 'dark' | 'system'
  language: string
  timezone: string
  notifications: NotificationSettings
  privacy: {
    shareUsageData: boolean
    allowAnalytics: boolean
  }
}

export interface PaymentMethod {
  id: string
  type: 'card' | 'paypal' | 'bank'
  last4?: string
  brand?: string
  expiryMonth?: number
  expiryYear?: number
  isDefault: boolean
  createdAt: Date
}

export interface Transaction {
  id: string
  type: 'purchase' | 'subscription' | 'refund'
  amount: number
  currency: string
  status: 'pending' | 'completed' | 'failed' | 'refunded'
  description: string
  createdAt: Date
  paymentMethod?: PaymentMethod
  items?: {
    adapterId: string
    adapterName: string
    price: number
  }[]
}

export interface AdapterReview {
  id: string
  adapterId: string
  userId: string
  userName: string
  userAvatar?: string
  rating: number
  title: string
  content: string
  helpful: number
  createdAt: Date
  verified: boolean
}

export interface AdapterStats {
  downloads: number
  rating: number
  reviews: number
  revenue: number
  activeUsers: number
  lastWeekGrowth: number
}
