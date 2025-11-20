// Generate an SVG favicon on the fly matching the theme
const faviconSvg = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <rect width="100" height="100" rx="25" fill="#2563eb"/>
    <text x="50" y="65" font-family="Arial, sans-serif" font-weight="bold" font-size="50" fill="white" text-anchor="middle">LC</text>
</svg>`;
const faviconUrl = "data:image/svg+xml;base64," + btoa(faviconSvg);
const link = document.createElement('link');
link.rel = 'icon';
link.href = faviconUrl;
document.head.appendChild(link);

// Toggle Contact Info with Smooth Animation
function toggleContact() {
    const contactInfo = document.getElementById('contact-info');

    if (contactInfo.style.maxHeight && contactInfo.style.maxHeight !== '0px') {
        contactInfo.style.maxHeight = '0px';
        contactInfo.style.opacity = '0';
        contactInfo.style.marginTop = '0px';
    } else {
        contactInfo.style.marginTop = '16px';
        contactInfo.style.maxHeight = contactInfo.scrollHeight + "px";
        contactInfo.style.opacity = '1';
    }
}

// --- Smart Theme Switching Logic Start ---
const themeBtn = document.getElementById('theme-toggle');
const icon = themeBtn.querySelector('i');

// 1. Function to apply theme switching visual effects
function applyTheme(isDark) {
    if (isDark) {
        document.documentElement.classList.add('dark');
        icon.classList.replace('fa-moon', 'fa-sun');
    } else {
        document.documentElement.classList.remove('dark');
        icon.classList.replace('fa-sun', 'fa-moon');
    }
}

// 2. Initialization: Determine which theme to use
function getPreferredTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        // If there is a manually saved preference, use it directly
        return savedTheme === 'dark';
    }
    // Otherwise, follow the system
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

// Apply immediately on page load
applyTheme(getPreferredTheme());

// 3. Click button event (core modification part)
themeBtn.addEventListener('click', () => {
    const isDarkNow = document.documentElement.classList.contains('dark');
    const targetTheme = !isDarkNow; // The target mode to switch to
    const systemIsDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    // Apply the new theme
    applyTheme(targetTheme);

    // Smart logic:
    // If the mode the user switches to is the same as the system's current mode, it means the user wants to "return to system"
    // We clear localStorage to let the website return to "auto mode"
    if (targetTheme === systemIsDark) {
        localStorage.removeItem('theme');
    } else {
        // Otherwise, the user is "going against the grain" (e.g., system is dark, user insists on light), then record the preference
        localStorage.setItem('theme', targetTheme ? 'dark' : 'light');
    }
});

// 4. Listen for system changes
// Only follow system changes when the user has not manually overridden (i.e., localStorage is empty)
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    if (!localStorage.getItem('theme')) {
        applyTheme(e.matches);
    }
});
// --- Smart Theme Switching Logic End ---

// BibTeX Copy
function copyBibtex(id) {
    const text = document.getElementById(id).value;
    navigator.clipboard.writeText(text).then(() => {
        const toast = document.getElementById('toast');
        toast.classList.remove('translate-x-20', 'opacity-0');
        setTimeout(() => {
            toast.classList.add('translate-x-20', 'opacity-0');
        }, 2500);
    });
}

// Auto-update date logic
const dateElement = document.getElementById('last-updated');
if (dateElement) {
    const date = new Date(document.lastModified);
    // Format: "Nov 20, 2025"
    dateElement.textContent = date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

// Dark mode extra styles - Optimized colors for softness
const style = document.createElement('style');
style.innerHTML = `
    .dark .bg-white { background-color: #1e293b !important; } 
    .dark .text-slate-900 { color: #ffffff !important; }
    .dark .text-slate-800 { color: #f8fafc !important; }
    .dark .text-slate-700 { color: #f1f5f9 !important; }
    .dark .text-slate-600 { color: #e2e8f0 !important; }
    .dark .text-slate-500 { color: #cbd5e1 !important; }
    .dark .border-slate-100 { border-color: #334155 !important; }
    .dark .border-slate-200 { border-color: #334155 !important; }
    .dark .bg-slate-50 { background-color: #0f172a !important; }
`;
document.head.appendChild(style);
