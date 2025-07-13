# Adaptrix Frontend

A modern, responsive Next.js frontend for the Adaptrix modular AI system.

## 🚀 Features

### 🎨 **Modern UI/UX**
- Beautiful, responsive design with Tailwind CSS
- Dark/light mode support
- Smooth animations with Framer Motion
- Glassmorphism effects and gradients

### 🔐 **Authentication**
- Clerk integration for secure authentication
- User management and profiles
- Protected routes and role-based access

### 💬 **Chat Interface**
- Real-time chat with AI
- Adapter selection and configuration
- Message history and session management
- RAG document integration
- Typing indicators and message actions

### 🛒 **Marketplace**
- Browse and purchase AI adapters
- Shopping cart and checkout flow
- Adapter reviews and ratings
- Category filtering and search

### 📊 **Dashboard**
- System overview and statistics
- Adapter management
- Document upload and management
- API key management
- Usage analytics

## 🛠️ Tech Stack

- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS + shadcn/ui components
- **Authentication**: Clerk
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Type Safety**: TypeScript

## 📁 Project Structure

```
src/website/
├── app/                    # Next.js app router pages
│   ├── (dashboard)/       # Protected dashboard routes
│   ├── marketplace/       # Public marketplace
│   └── layout.tsx         # Root layout
├── components/            # Reusable components
│   ├── ui/               # Base UI components (shadcn/ui)
│   ├── layout/           # Layout components
│   ├── sections/         # Landing page sections
│   ├── chat/             # Chat interface components
│   ├── dashboard/        # Dashboard components
│   └── marketplace/      # Marketplace components
├── lib/                  # Utilities and types
│   ├── utils.ts          # Helper functions
│   └── types.ts          # TypeScript types
└── styles/               # Global styles
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ 
- npm/yarn/pnpm

### Installation

1. **Clone and navigate to the frontend directory**:
   ```bash
   cd src/website
   ```

2. **Install dependencies**:
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env.local
   ```
   
   Update the environment variables:
   ```env
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
   CLERK_SECRET_KEY=your_clerk_secret_key
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

4. **Run the development server**:
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

5. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🔧 Configuration

### Clerk Authentication

1. Create a Clerk account at [clerk.com](https://clerk.com)
2. Create a new application
3. Copy the publishable key and secret key to your `.env.local` file
4. Configure sign-in/sign-up options in the Clerk dashboard

### API Integration

The frontend is designed to work with the Adaptrix backend API. Update the `NEXT_PUBLIC_API_URL` environment variable to point to your backend server.

## 📱 Pages and Features

### 🏠 **Landing Page** (`/`)
- Hero section with animated elements
- Feature showcase
- Adapter marketplace preview
- Technology stack overview
- Testimonials and pricing
- Call-to-action sections

### 📊 **Dashboard** (`/dashboard`)
- System overview and statistics
- Quick actions and recent activity
- System status monitoring

### 💬 **Chat Interface** (`/dashboard/chat`)
- Real-time AI chat
- Adapter selection and switching
- Chat history and sessions
- Settings panel for model configuration
- RAG document integration

### 🛒 **Marketplace** (`/marketplace`)
- Browse available adapters
- Category filtering and search
- Shopping cart functionality
- Adapter details and reviews
- Purchase flow (frontend only)

### ⚙️ **Settings Pages**
- `/dashboard/adapters` - Manage purchased adapters
- `/dashboard/documents` - Upload and manage RAG documents
- `/dashboard/api-keys` - API key management
- `/dashboard/billing` - Billing and subscription management
- `/dashboard/settings` - User preferences and configuration

## 🎨 Design System

### Colors
- **Primary**: Blue to Purple gradient
- **Secondary**: Muted grays
- **Accent**: Various gradients for categories
- **Success**: Green tones
- **Warning**: Orange/Yellow tones
- **Error**: Red tones

### Typography
- **Font**: Inter (Google Fonts)
- **Headings**: Bold, gradient text effects
- **Body**: Regular weight, good contrast
- **Code**: Monospace for technical content

### Components
- Built with shadcn/ui for consistency
- Custom components for specific features
- Responsive design for all screen sizes
- Accessibility-first approach

## 🔄 State Management

The frontend uses React's built-in state management with:
- `useState` for local component state
- `useEffect` for side effects
- Context API for global state (when needed)
- Local storage for user preferences

## 🚀 Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Add environment variables in Vercel dashboard
4. Deploy automatically on push

### Other Platforms

The app can be deployed to any platform that supports Next.js:
- Netlify
- AWS Amplify
- Railway
- DigitalOcean App Platform

## 🧪 Development

### Adding New Components

1. Create component in appropriate directory
2. Follow naming conventions (PascalCase)
3. Include TypeScript types
4. Add to index files if needed

### Styling Guidelines

1. Use Tailwind CSS classes
2. Follow mobile-first responsive design
3. Use CSS variables for theming
4. Maintain consistent spacing and typography

### Code Quality

- TypeScript for type safety
- ESLint for code linting
- Prettier for code formatting
- Consistent naming conventions

## 📝 Notes

- This is a frontend-only implementation
- Backend API calls are mocked for demonstration
- Real backend integration requires updating API endpoints
- Authentication flows are fully functional with Clerk
- Shopping cart and checkout are UI-only (no payment processing)

## 🤝 Contributing

1. Follow the existing code style
2. Add TypeScript types for new features
3. Test responsive design on multiple screen sizes
4. Ensure accessibility standards are met

## 📄 License

This project is part of the Adaptrix system and follows the same license terms.
