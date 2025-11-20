# Langqing Cui's Personal Website

My personal academic website built with React + Vite + Tailwind CSS.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📦 Tech Stack

- **React** 19.2.0 - UI framework
- **Vite** 7.2.4 - Build tool & dev server
- **Tailwind CSS** 3.4.17 - Styling
- **GitHub Pages** - Hosting

## 🛠️ Development

The site will be available at `http://localhost:5173` when running `npm run dev`.

### Project Structure

```
src/
├── App.jsx                 # Main application
├── main.jsx               # Entry point
├── index.css              # Global styles
├── components/            # React components
│   ├── Navbar.jsx
│   ├── Hero.jsx
│   ├── News.jsx
│   ├── Publications.jsx
│   ├── Experience.jsx
│   ├── Education.jsx
│   └── Footer.jsx
└── hooks/
    └── useTheme.js        # Dark mode logic
```

## 🌐 Deployment

This site is automatically deployed to GitHub Pages via GitHub Actions when pushing to the `main` branch.

**Setup GitHub Pages:**

1. Go to **Settings** → **Pages**
2. Set **Source** to "GitHub Actions"
3. Push to `main` branch to trigger deployment

## 📄 License

© 2025 Langqing Cui. All rights reserved.
