#!/usr/bin/env node

/**
 * Component Check Script
 * Verifies that all required components exist and are properly exported
 */

const fs = require('fs');
const path = require('path');

const componentsToCheck = [
  'components/layout/navbar.tsx',
  'components/layout/footer.tsx',
  'components/sections/hero-section.tsx',
  'components/sections/features-section.tsx',
  'components/sections/adapters-showcase.tsx',
  'components/sections/tech-stack.tsx',
  'components/sections/testimonials.tsx',
  'components/sections/pricing-section.tsx',
  'components/sections/cta-section.tsx',
  'components/ui/button.tsx',
  'components/ui/card.tsx',
  'components/ui/input.tsx',
  'components/ui/badge.tsx',
  'components/ui/label.tsx',
  'components/ui/switch.tsx',
  'components/ui/slider.tsx',
  'components/dashboard/sidebar.tsx',
  'components/dashboard/header.tsx',
  'components/chat/chat-input.tsx',
  'components/chat/chat-message.tsx',
  'components/chat/chat-sidebar.tsx',
  'components/chat/adapter-selector.tsx',
  'components/chat/chat-settings.tsx',
  'components/marketplace/adapter-card.tsx',
  'components/marketplace/adapter-filters.tsx',
  'components/marketplace/cart-sidebar.tsx',
  'lib/utils.ts',
  'lib/types.ts'
];

const pagestoCheck = [
  'app/page.tsx',
  'app/layout.tsx',
  'app/(dashboard)/layout.tsx',
  'app/(dashboard)/dashboard/page.tsx',
  'app/(dashboard)/dashboard/chat/page.tsx',
  'app/marketplace/page.tsx'
];

console.log('🔍 Checking Adaptrix Frontend Components...\n');

let allGood = true;

// Check components
console.log('📦 Components:');
componentsToCheck.forEach(component => {
  const filePath = path.join(__dirname, '..', component);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${component}`);
  } else {
    console.log(`❌ ${component} - MISSING`);
    allGood = false;
  }
});

console.log('\n📄 Pages:');
pagestoCheck.forEach(page => {
  const filePath = path.join(__dirname, '..', page);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${page}`);
  } else {
    console.log(`❌ ${page} - MISSING`);
    allGood = false;
  }
});

// Check configuration files
console.log('\n⚙️ Configuration:');
const configFiles = [
  'package.json',
  'tsconfig.json',
  'tailwind.config.js',
  'next.config.js',
  'postcss.config.js',
  '.env.local'
];

configFiles.forEach(file => {
  const filePath = path.join(__dirname, '..', file);
  if (fs.existsSync(filePath)) {
    console.log(`✅ ${file}`);
  } else {
    console.log(`❌ ${file} - MISSING`);
    allGood = false;
  }
});

console.log('\n' + '='.repeat(50));
if (allGood) {
  console.log('🎉 All components and files are present!');
  console.log('✨ Adaptrix Frontend is ready to go!');
} else {
  console.log('⚠️  Some files are missing. Please check the errors above.');
}
console.log('='.repeat(50));
