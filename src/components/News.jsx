export default function News() {
  const newsItems = [
    {
      date: 'Sep 2025',
      text: (
        <>
          Joined <b>FlexML Lab</b> at <b>KAIST EE</b>!
        </>
      ),
      color: 'primary',
    },
    {
      date: 'Jun 2025',
      text: (
        <>
          Graduated from <b>Beijing University of Technology</b> 🎉
        </>
      ),
      color: 'red',
    },
  ];

  return (
    <section
      id="news"
      className="animate-fade-in-up scroll-mt-14 md:scroll-mt-20"
      style={{ animationDelay: '0.1s' }}
    >
      <div className="flex items-center gap-3 mb-8">
        <div className="p-2 bg-red-100 dark:bg-red-900/20 rounded-lg text-red-600 dark:text-red-400">
          <i className="fas fa-bullhorn"></i>
        </div>
        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Latest News</h2>
      </div>

      <div className="relative border-l-2 border-slate-200 dark:border-slate-700 ml-3 space-y-8 py-2">
        {newsItems.map((item, index) => (
          <div key={index} className="relative pl-8 group">
            <div
              className={`absolute -left-[9px] top-2 w-4 h-4 bg-white dark:bg-slate-800 border-2 border-${item.color}-500 rounded-full group-hover:scale-125 transition-transform`}
            ></div>
            <div className="flex flex-col sm:flex-row gap-1 sm:gap-4 sm:items-baseline">
              <span className="font-mono text-sm font-bold text-slate-400">{item.date}</span>
              <span className="text-slate-700 dark:text-slate-300">{item.text}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
