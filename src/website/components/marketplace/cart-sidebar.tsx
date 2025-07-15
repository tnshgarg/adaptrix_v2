'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, ShoppingCart, Trash2, CreditCard, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Adapter } from '@/lib/types'
import { formatPrice, cn } from '@/lib/utils'

interface CartSidebarProps {
  isOpen: boolean
  onClose: () => void
  cartItems: Adapter[]
  onRemoveItem: (adapterId: string) => void
}

export function CartSidebar({ isOpen, onClose, cartItems, onRemoveItem }: CartSidebarProps) {
  const [isCheckingOut, setIsCheckingOut] = useState(false)

  const subtotal = cartItems.reduce((sum, item) => sum + item.price, 0)
  const tax = subtotal * 0.08 // 8% tax
  const total = subtotal + tax

  const handleCheckout = async () => {
    setIsCheckingOut(true)
    // Simulate checkout process
    await new Promise(resolve => setTimeout(resolve, 2000))
    setIsCheckingOut(false)
    // In a real app, this would redirect to payment processing
    alert('Checkout functionality would be implemented here!')
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-50"
            onClick={onClose}
          />

          {/* Sidebar */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-96 bg-background border-l shadow-xl z-50 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center space-x-2">
                <ShoppingCart className="w-5 h-5" />
                <h2 className="text-lg font-semibold">Shopping Cart</h2>
                <Badge variant="secondary">{cartItems.length}</Badge>
              </div>
              <Button variant="ghost" size="sm" onClick={onClose}>
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Cart Items */}
            <div className="flex-1 overflow-y-auto p-6">
              {cartItems.length === 0 ? (
                <div className="text-center py-12">
                  <ShoppingCart className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-2">Your cart is empty</h3>
                  <p className="text-muted-foreground mb-4">
                    Add some adapters to get started
                  </p>
                  <Button onClick={onClose}>
                    Continue Shopping
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  {cartItems.map((item) => (
                    <motion.div
                      key={item.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      layout
                    >
                      <Card>
                        <CardContent className="p-4">
                          <div className="flex items-start space-x-3">
                            <div className="w-12 h-12 rounded-lg bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 flex items-center justify-center shrink-0">
                              <span className="text-sm font-semibold text-blue-600">
                                {item.category.charAt(0).toUpperCase()}
                              </span>
                            </div>
                            
                            <div className="flex-1 min-w-0">
                              <h4 className="font-medium text-sm line-clamp-1">
                                {item.name}
                              </h4>
                              <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
                                {item.description}
                              </p>
                              <div className="flex items-center justify-between mt-2">
                                <Badge variant="outline" className="text-xs">
                                  {item.category}
                                </Badge>
                                <div className="text-sm font-semibold">
                                  {formatPrice(item.price)}
                                </div>
                              </div>
                            </div>

                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => onRemoveItem(item.id)}
                              className="shrink-0"
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>

            {/* Footer */}
            {cartItems.length > 0 && (
              <div className="border-t p-6 space-y-4">
                {/* Summary */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Subtotal</span>
                    <span>{formatPrice(subtotal)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Tax</span>
                    <span>{formatPrice(tax)}</span>
                  </div>
                  <div className="flex justify-between font-semibold border-t pt-2">
                    <span>Total</span>
                    <span>{formatPrice(total)}</span>
                  </div>
                </div>

                {/* Checkout Button */}
                <Button
                  className="w-full"
                  size="lg"
                  onClick={handleCheckout}
                  disabled={isCheckingOut}
                >
                  {isCheckingOut ? (
                    <div className="flex items-center space-x-2">
                      <div className="spinner" />
                      <span>Processing...</span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-2">
                      <CreditCard className="w-4 h-4" />
                      <span>Checkout</span>
                      <ArrowRight className="w-4 h-4" />
                    </div>
                  )}
                </Button>

                <p className="text-xs text-muted-foreground text-center">
                  Secure checkout powered by Stripe
                </p>
              </div>
            )}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
