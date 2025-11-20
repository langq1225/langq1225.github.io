import Navbar from './components/Navbar';
import Hero from './components/Hero';
import News from './components/News';
import Publications from './components/Publications';
import Experience from './components/Experience';
import Education from './components/Education';
import Footer from './components/Footer';

function App() {
  return (
    <>
      <Navbar />
      
      {/* Main Container */}
      <main className="max-w-5xl mx-auto px-6 pt-24 pb-20 space-y-20">
        <Hero />
        <News />
        <Publications />
        
        {/* Grid: Education & Experience */}
        <div
          className="grid md:grid-cols-2 gap-8 animate-fade-in-up"
          style={{ animationDelay: '0.3s' }}
        >
          <Experience />
          <Education />
        </div>

        <Footer />
      </main>

      {/* Toast Notification */}
      <div
        id="toast"
        className="fixed top-24 right-6 transform translate-x-20 opacity-0 transition-all duration-300 z-50 bg-slate-800 text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3"
      >
        <i className="fas fa-check-circle text-green-400"></i>
        <span className="text-sm font-medium">Copied to clipboard!</span>
      </div>
    </>
  );
}

export default App;
