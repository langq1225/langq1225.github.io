import { useEffect, useState } from 'react';

export function useTheme() {
  const [isDark, setIsDark] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme === 'dark';
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    // Apply theme to DOM
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }

    // Generate and set favicon
    const faviconSvg = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect width="100" height="100" rx="25" fill="#2563eb"/>
        <text x="50" y="65" font-family="Arial, sans-serif" font-weight="bold" font-size="50" fill="white" text-anchor="middle">LC</text>
      </svg>`;
    const faviconUrl = "data:image/svg+xml;base64," + btoa(faviconSvg);
    let link = document.querySelector("link[rel~='icon']");
    if (!link) {
      link = document.createElement('link');
      link.rel = 'icon';
      document.head.appendChild(link);
    }
    link.href = faviconUrl;
  }, [isDark]);

  useEffect(() => {
    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e) => {
      if (!localStorage.getItem('theme')) {
        setIsDark(e.matches);
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const toggleTheme = () => {
    const newTheme = !isDark;
    const systemIsDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    setIsDark(newTheme);
    
    // Smart logic: if switching to system theme, remove override
    if (newTheme === systemIsDark) {
      localStorage.removeItem('theme');
    } else {
      localStorage.setItem('theme', newTheme ? 'dark' : 'light');
    }
  };

  return { isDark, toggleTheme };
}
