export default function Education() {
  const educationItems = [
    {
      id: 1,
      school: 'KAIST',
      degree: 'M.S. in Electrical Engineering',
      period: 'Sep 2025 - Present',
      active: true,
    },
    {
      id: 2,
      school: 'Beijing University of Technology',
      degree: 'B.Eng. in Computer Science and Technology',
      period: 'Sep 2021 - Jun 2025',
      active: false,
    },
  ];

  return (
    <section className="bg-white dark:bg-slate-800/80 p-8 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-emerald-100 dark:bg-emerald-900/20 rounded-lg text-emerald-600 dark:text-emerald-400">
          <i className="fas fa-graduation-cap"></i>
        </div>
        <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">Education</h2>
      </div>

      <div className="space-y-6">
        {educationItems.map((item) => (
          <div
            key={item.id}
            className="relative pl-6 border-l border-slate-200 dark:border-slate-700"
          >
            <div
              className={`absolute -left-[5px] top-2 w-2.5 h-2.5 rounded-full ${
                item.active
                  ? 'bg-emerald-500'
                  : 'bg-slate-300 dark:bg-slate-600'
              }`}
            ></div>
            <h4 className="font-bold text-slate-900 dark:text-slate-100">{item.school}</h4>
            <div className="text-sm text-slate-600 dark:text-slate-300">{item.degree}</div>
            <div className="text-xs text-slate-400 mt-1">{item.period}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
