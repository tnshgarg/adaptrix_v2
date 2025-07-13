import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { ClerkProvider } from '@clerk/nextjs'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Adaptrix - The World\'s First Modular AI System',
  description: 'Revolutionary modular AI system with dynamic LoRA adapter composition, automatic task classification, and RAG integration.',
  keywords: ['AI', 'Machine Learning', 'Modular AI', 'LoRA', 'RAG', 'Adaptrix'],
  authors: [{ name: 'Adaptrix Team' }],
  openGraph: {
    title: 'Adaptrix - The World\'s First Modular AI System',
    description: 'Revolutionary modular AI system with dynamic LoRA adapter composition, automatic task classification, and RAG integration.',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Adaptrix - The World\'s First Modular AI System',
    description: 'Revolutionary modular AI system with dynamic LoRA adapter composition, automatic task classification, and RAG integration.',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ClerkProvider>
      <html lang="en" suppressHydrationWarning>
        <body className={inter.className}>
          <div className="min-h-screen bg-background">
            {children}
          </div>
        </body>
      </html>
    </ClerkProvider>
  )
}
