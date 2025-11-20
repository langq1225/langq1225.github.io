import { useTheme } from '../hooks/useTheme';

export default function Navbar() {
  const { isDark, toggleTheme } = useTheme();

  return (
    <nav className="fixed w-full z-50 top-0 glass shadow-sm transition-all duration-300">
      <div className="max-w-5xl mx-auto px-6 py-3 flex justify-between items-center">
        {/* Logo & Title Section */}
        <a href="#" className="flex items-center gap-3 group">
          {/* Flat Modern Logo */}
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-primary-500 to-indigo-600 flex items-center justify-center shadow-md transition-transform duration-300 group-hover:scale-110 group-hover:shadow-lg group-hover:rotate-3">
            <span className="text-white font-bold text-sm tracking-tighter leading-none">
              LC
            </span>
          </div>
          {/* Text */}
          <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-indigo-600 dark:from-primary-400 dark:to-indigo-400">
            Langqing Cui
          </span>
        </a>

        <div className="hidden md:flex space-x-8 text-sm font-medium text-slate-600 dark:text-slate-300">
          <a href="#about" className="nav-link hover:text-primary-600 transition-colors">
            About
          </a>
          <a href="#news" className="nav-link hover:text-primary-600 transition-colors">
            News
          </a>
          <a href="#publications" className="nav-link hover:text-primary-600 transition-colors">
            Publications
          </a>
          <a href="#experience" className="nav-link hover:text-primary-600 transition-colors">
            Experience
          </a>
        </div>

        {/* Dark Mode Toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors text-slate-600 dark:text-slate-400"
          aria-label="Toggle Dark Mode"
        >
          <i className={`fas fa-${isDark ? 'sun' : 'moon'}`}></i>
        </button>
      </div>
    </nav>
  );
}
