export default function Publications() {
  return (
    <section
      id="publications"
      className="animate-fade-in-up scroll-mt-14 md:scroll-mt-20"
      style={{ animationDelay: '0.2s' }}
    >
      <div className="flex items-center gap-3 mb-8">
        <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg text-primary-600 dark:text-primary-400">
          <i className="fas fa-book-open"></i>
        </div>
        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Publications</h2>
      </div>

      <div className="bg-white dark:bg-slate-800/80 p-8 rounded-xl shadow-sm border border-slate-100 dark:border-slate-700 text-center">
        <div className="inline-block p-4 rounded-full bg-slate-50 dark:bg-slate-700/50 mb-4">
          <i className="fas fa-rocket text-2xl text-slate-400"></i>
        </div>
        <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-2">
          Research in Progress
        </h3>
        <p className="text-slate-500 dark:text-slate-400 max-w-lg mx-auto">
          Unfortunately, 0 papers have been published... yet!
        </p>
      </div>
    </section>
  );
}
