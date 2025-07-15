'use client'

import { motion } from 'framer-motion'
import { 
  MessageSquare, 
  Puzzle, 
  FileText, 
  Zap, 
  TrendingUp, 
  Clock,
  Brain,
  Database
} from 'lucide-react'
import Link from 'next/link'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export default function DashboardPage() {
  const stats = [
    {
      title: "Total Chats",
      value: "24",
      change: "+12%",
      icon: MessageSquare,
      color: "text-blue-600"
    },
    {
      title: "Active Adapters",
      value: "8",
      change: "+2",
      icon: Puzzle,
      color: "text-purple-600"
    },
    {
      title: "Documents",
      value: "156",
      change: "+23",
      icon: FileText,
      color: "text-green-600"
    },
    {
      title: "API Calls",
      value: "1,234",
      change: "+45%",
      icon: Zap,
      color: "text-orange-600"
    }
  ]

  const recentActivity = [
    {
      type: "chat",
      title: "Code generation session",
      description: "Generated Python function for data processing",
      time: "2 minutes ago",
      adapter: "Code Adapter"
    },
    {
      type: "adapter",
      title: "Legal Adapter installed",
      description: "Successfully installed legal document analyzer",
      time: "1 hour ago",
      adapter: "Legal Adapter"
    },
    {
      type: "document",
      title: "Documents uploaded",
      description: "Added 5 new documents to RAG system",
      time: "3 hours ago",
      adapter: null
    }
  ]

  const quickActions = [
    {
      title: "Start New Chat",
      description: "Begin a conversation with AI",
      href: "/dashboard/chat",
      icon: MessageSquare,
      color: "from-blue-500 to-cyan-500"
    },
    {
      title: "Browse Adapters",
      description: "Discover new AI capabilities",
      href: "/marketplace",
      icon: Puzzle,
      color: "from-purple-500 to-pink-500"
    },
    {
      title: "Upload Documents",
      description: "Add knowledge to RAG system",
      href: "/dashboard/documents",
      icon: FileText,
      color: "from-green-500 to-emerald-500"
    }
  ]

  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-3xl font-bold">Welcome back!</h1>
        <p className="text-muted-foreground mt-2">
          Here's what's happening with your Adaptrix system today.
        </p>
      </motion.div>

      {/* Stats Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {stats.map((stat, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {stat.title}
              </CardTitle>
              <stat.icon className={`w-4 h-4 ${stat.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">
                <span className="text-green-600">{stat.change}</span> from last month
              </p>
            </CardContent>
          </Card>
        ))}
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="lg:col-span-2"
        >
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
              <CardDescription>
                Get started with common tasks
              </CardDescription>
            </CardHeader>
            <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {quickActions.map((action, index) => (
                <Link key={index} href={action.href}>
                  <Card className="group hover:shadow-md transition-all duration-200 cursor-pointer">
                    <CardContent className="p-6 text-center">
                      <div className={`w-12 h-12 mx-auto mb-4 rounded-lg bg-gradient-to-r ${action.color} flex items-center justify-center group-hover:scale-110 transition-transform`}>
                        <action.icon className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="font-semibold mb-2">{action.title}</h3>
                      <p className="text-sm text-muted-foreground">
                        {action.description}
                      </p>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </CardContent>
          </Card>
        </motion.div>

        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>
                Your latest interactions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {recentActivity.map((activity, index) => (
                <div key={index} className="flex items-start space-x-3">
                  <div className="w-2 h-2 rounded-full bg-blue-600 mt-2" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium">{activity.title}</p>
                    <p className="text-xs text-muted-foreground">
                      {activity.description}
                    </p>
                    <div className="flex items-center space-x-2 mt-1">
                      <Clock className="w-3 h-3 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">
                        {activity.time}
                      </span>
                      {activity.adapter && (
                        <>
                          <span className="text-xs text-muted-foreground">•</span>
                          <span className="text-xs text-blue-600">
                            {activity.adapter}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <Card>
          <CardHeader>
            <CardTitle>System Status</CardTitle>
            <CardDescription>
              Current status of your Adaptrix components
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <div>
                  <p className="text-sm font-medium">Base Model</p>
                  <p className="text-xs text-muted-foreground">Qwen-3 1.7B Online</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <div>
                  <p className="text-sm font-medium">MoE Classifier</p>
                  <p className="text-xs text-muted-foreground">96.5% Accuracy</p>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <div>
                  <p className="text-sm font-medium">RAG System</p>
                  <p className="text-xs text-muted-foreground">156 Documents Indexed</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
