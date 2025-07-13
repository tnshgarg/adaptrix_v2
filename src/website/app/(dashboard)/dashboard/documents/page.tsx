'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  FileText, 
  Upload, 
  Trash2, 
  Search, 
  Filter,
  Download,
  Eye,
  Calendar,
  FileIcon,
  Plus
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'

export default function DocumentsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  
  const documents = [
    {
      id: '1',
      name: 'API Documentation.pdf',
      type: 'pdf',
      size: '2.4 MB',
      uploadDate: '2024-01-20',
      status: 'indexed',
      chunks: 45,
      category: 'technical'
    },
    {
      id: '2',
      name: 'Legal Guidelines.docx',
      type: 'docx',
      size: '1.8 MB',
      uploadDate: '2024-01-18',
      status: 'processing',
      chunks: 32,
      category: 'legal'
    },
    {
      id: '3',
      name: 'Company Policies.txt',
      type: 'txt',
      size: '156 KB',
      uploadDate: '2024-01-15',
      status: 'indexed',
      chunks: 12,
      category: 'policy'
    },
    {
      id: '4',
      name: 'Research Paper.pdf',
      type: 'pdf',
      size: '3.2 MB',
      uploadDate: '2024-01-12',
      status: 'failed',
      chunks: 0,
      category: 'research'
    }
  ]

  const filteredDocuments = documents.filter(doc =>
    doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    doc.category.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'indexed': return 'bg-green-600'
      case 'processing': return 'bg-yellow-600'
      case 'failed': return 'bg-red-600'
      default: return 'bg-gray-600'
    }
  }

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'pdf': return '📄'
      case 'docx': return '📝'
      case 'txt': return '📋'
      default: return '📄'
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Documents</h1>
          <p className="text-muted-foreground mt-2">
            Manage your RAG document collection for enhanced AI responses
          </p>
        </div>
        <Button>
          <Plus className="w-4 h-4 mr-2" />
          Upload Documents
        </Button>
      </div>

      {/* Search and Filters */}
      <div className="flex items-center space-x-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        <Button variant="outline">
          <Filter className="w-4 h-4 mr-2" />
          Filter
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{documents.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Indexed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {documents.filter(d => d.status === 'indexed').length}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Chunks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {documents.reduce((sum, doc) => sum + doc.chunks, 0)}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">7.6 MB</div>
          </CardContent>
        </Card>
      </div>

      {/* Upload Area */}
      <Card className="border-2 border-dashed border-muted-foreground/25 hover:border-muted-foreground/50 transition-colors">
        <CardContent className="p-8">
          <div className="text-center">
            <Upload className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">Upload Documents</h3>
            <p className="text-muted-foreground mb-4">
              Drag and drop files here, or click to browse
            </p>
            <Button>Choose Files</Button>
            <p className="text-xs text-muted-foreground mt-2">
              Supports PDF, DOCX, TXT files up to 10MB each
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Documents List */}
      <div className="space-y-4">
        {filteredDocuments.map((document, index) => (
          <motion.div
            key={document.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <Card>
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    <div className="w-12 h-12 bg-muted rounded-lg flex items-center justify-center text-2xl">
                      {getFileIcon(document.type)}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="text-lg font-semibold">{document.name}</h3>
                        <Badge 
                          variant="secondary"
                          className={getStatusColor(document.status)}
                        >
                          {document.status}
                        </Badge>
                        <Badge variant="outline">{document.category}</Badge>
                      </div>
                      
                      <div className="flex items-center space-x-6 text-sm text-muted-foreground">
                        <span>{document.size}</span>
                        <div className="flex items-center space-x-1">
                          <Calendar className="w-4 h-4" />
                          <span>Uploaded {document.uploadDate}</span>
                        </div>
                        <span>{document.chunks} chunks</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Button variant="outline" size="sm">
                      <Eye className="w-4 h-4 mr-2" />
                      View
                    </Button>
                    <Button variant="outline" size="sm">
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </Button>
                    <Button variant="outline" size="sm">
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {filteredDocuments.length === 0 && (
        <div className="text-center py-12">
          <FileText className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium mb-2">No documents found</h3>
          <p className="text-muted-foreground mb-4">
            Upload your first document to get started with RAG.
          </p>
          <Button>Upload Document</Button>
        </div>
      )}
    </div>
  )
}
