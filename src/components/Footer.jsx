import { useEffect, useState } from 'react';

export default function Footer() {
  const [lastUpdated, setLastUpdated] = useState('');

  useEffect(() => {
    const date = new Date(document.lastModified);
    setLastUpdated(
      date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
    );
  }, []);

  return (
    <footer className="text-center py-8 text-slate-500 text-sm">
      <p>
        © 2025 Langqing Cui. Built with
        <i className="fas fa-heart text-red-400 mx-1"></i>
        and Tailwind CSS.
      </p>
      <p className="text-xs text-slate-400 dark:text-slate-600 mt-2">
        Last updated: <span>{lastUpdated}</span>
      </p>
    </footer>
  );
}
