'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  CreditCard, 
  Download, 
  Calendar, 
  DollarSign,
  TrendingUp,
  Package,
  Check,
  Zap
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'

export default function BillingPage() {
  const currentPlan = {
    name: 'Pro',
    price: 29,
    period: 'month',
    features: [
      'Unlimited adapters',
      '100GB document storage',
      'Priority support',
      'Advanced analytics',
      'Custom integrations'
    ],
    usage: {
      adapters: { used: 8, limit: 'unlimited' },
      storage: { used: 45, limit: 100 },
      apiCalls: { used: 15420, limit: 50000 }
    }
  }

  const plans = [
    {
      name: 'Starter',
      price: 0,
      period: 'month',
      description: 'Perfect for getting started',
      features: [
        '3 adapters',
        '1GB document storage',
        'Community support',
        'Basic analytics'
      ],
      popular: false
    },
    {
      name: 'Pro',
      price: 29,
      period: 'month',
      description: 'Best for professionals',
      features: [
        'Unlimited adapters',
        '100GB document storage',
        'Priority support',
        'Advanced analytics',
        'Custom integrations'
      ],
      popular: true
    },
    {
      name: 'Enterprise',
      price: 99,
      period: 'month',
      description: 'For large organizations',
      features: [
        'Everything in Pro',
        'Unlimited storage',
        'Dedicated support',
        'Custom deployment',
        'SLA guarantee'
      ],
      popular: false
    }
  ]

  const invoices = [
    {
      id: 'INV-001',
      date: '2024-01-01',
      amount: 29.00,
      status: 'paid',
      plan: 'Pro Monthly'
    },
    {
      id: 'INV-002',
      date: '2023-12-01',
      amount: 29.00,
      status: 'paid',
      plan: 'Pro Monthly'
    },
    {
      id: 'INV-003',
      date: '2023-11-01',
      amount: 29.00,
      status: 'paid',
      plan: 'Pro Monthly'
    }
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Billing & Usage</h1>
          <p className="text-muted-foreground mt-2">
            Manage your subscription and view usage statistics
          </p>
        </div>
        <Button>
          <CreditCard className="w-4 h-4 mr-2" />
          Update Payment Method
        </Button>
      </div>

      {/* Current Plan */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center space-x-2">
                <span>Current Plan: {currentPlan.name}</span>
                <Badge className="bg-blue-600">Active</Badge>
              </CardTitle>
              <CardDescription>
                ${currentPlan.price}/{currentPlan.period} • Next billing: February 1, 2024
              </CardDescription>
            </div>
            <Button variant="outline">Change Plan</Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Adapters</span>
                <span className="text-sm text-muted-foreground">
                  {currentPlan.usage.adapters.used} / {currentPlan.usage.adapters.limit}
                </span>
              </div>
              <Progress value={0} className="h-2" />
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">Storage</span>
                <span className="text-sm text-muted-foreground">
                  {currentPlan.usage.storage.used}GB / {currentPlan.usage.storage.limit}GB
                </span>
              </div>
              <Progress value={(currentPlan.usage.storage.used / currentPlan.usage.storage.limit) * 100} className="h-2" />
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">API Calls</span>
                <span className="text-sm text-muted-foreground">
                  {currentPlan.usage.apiCalls.used.toLocaleString()} / {currentPlan.usage.apiCalls.limit.toLocaleString()}
                </span>
              </div>
              <Progress value={(currentPlan.usage.apiCalls.used / currentPlan.usage.apiCalls.limit) * 100} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Usage Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">This Month</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$29.00</div>
            <p className="text-xs text-muted-foreground">Current billing cycle</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">API Calls</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">15.4K</div>
            <p className="text-xs text-muted-foreground">This month</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">45GB</div>
            <p className="text-xs text-muted-foreground">45% of limit</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Active Adapters</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">8</div>
            <p className="text-xs text-muted-foreground">Installed</p>
          </CardContent>
        </Card>
      </div>

      {/* Available Plans */}
      <div>
        <h2 className="text-2xl font-bold mb-6">Available Plans</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
            >
              <Card className={`relative ${plan.popular ? 'border-blue-500 shadow-lg' : ''}`}>
                {plan.popular && (
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <Badge className="bg-blue-600">Most Popular</Badge>
                  </div>
                )}
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{plan.name}</span>
                    {plan.name === currentPlan.name && <Check className="w-5 h-5 text-green-600" />}
                  </CardTitle>
                  <CardDescription>{plan.description}</CardDescription>
                  <div className="text-3xl font-bold">
                    ${plan.price}
                    <span className="text-lg font-normal text-muted-foreground">/{plan.period}</span>
                  </div>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2 mb-6">
                    {plan.features.map((feature, i) => (
                      <li key={i} className="flex items-center space-x-2">
                        <Check className="w-4 h-4 text-green-600" />
                        <span className="text-sm">{feature}</span>
                      </li>
                    ))}
                  </ul>
                  <Button 
                    className="w-full" 
                    variant={plan.name === currentPlan.name ? "outline" : "default"}
                    disabled={plan.name === currentPlan.name}
                  >
                    {plan.name === currentPlan.name ? 'Current Plan' : 'Upgrade'}
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Billing History */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Billing History</CardTitle>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Download All
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {invoices.map((invoice) => (
              <div key={invoice.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="w-10 h-10 bg-muted rounded-lg flex items-center justify-center">
                    <DollarSign className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="font-medium">{invoice.plan}</div>
                    <div className="text-sm text-muted-foreground">
                      {invoice.id} • {invoice.date}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="font-medium">${invoice.amount.toFixed(2)}</div>
                    <Badge variant={invoice.status === 'paid' ? 'default' : 'secondary'}>
                      {invoice.status}
                    </Badge>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
