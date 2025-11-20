export default function Experience() {
  const experiences = [
    {
      id: 1,
      logo: '/images/kaist_ee_logo.png',
      fallback: 'K',
      title: 'School of Electrical Engineering, KAIST',
      role: 'Research Assistant',
      period: 'Sep 2025 - Present',
      description: (
        <>
          Supervised by Prof.{' '}
          <a
            href="https://insuhan.github.io/"
            className="text-primary-600 hover:underline decoration-2 underline-offset-2"
          >
            Insu Han
          </a>
          .
        </>
      ),
    },
    {
      id: 2,
      logo: '/images/bjut_logo.png',
      fallback: 'B',
      title: 'Beijing University of Technology',
      role: 'Research Assistant',
      period: 'Nov 2024 - Jun 2025',
      description: (
        <>
          Supervised by Prof.{' '}
          <a
            href="https://scholar.google.com/citations?user=jAc5SHAAAAAJ&hl=en-US"
            className="text-primary-600 hover:underline decoration-2 underline-offset-2"
          >
            Tongtong Yuan
          </a>
          .
        </>
      ),
    },
  ];

  return (
    <section
      id="experience"
      className="bg-white dark:bg-slate-800/80 p-8 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700 scroll-mt-14 md:scroll-mt-20"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-purple-100 dark:bg-purple-900/20 rounded-lg text-purple-600 dark:text-purple-400">
          <i className="fas fa-briefcase"></i>
        </div>
        <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">Experience</h2>
      </div>

      <div className="space-y-6">
        {experiences.map((exp) => (
          <div key={exp.id} className="flex gap-4 items-start">
            {/* Logo/Icon Container */}
            <div className="w-10 h-10 rounded bg-slate-50 dark:bg-slate-700/50 flex items-center justify-center flex-shrink-0 overflow-hidden relative border border-slate-100 dark:border-slate-600">
              {/* Fallback Text */}
              <span className="text-xs font-bold text-slate-500 dark:text-slate-400">
                {exp.fallback}
              </span>

              {/* Logo Image */}
              <img
                src={exp.logo}
                alt={`${exp.title} Logo`}
                loading="lazy"
                className="absolute inset-0 w-full h-full object-contain bg-white dark:bg-slate-800 p-0.5"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            </div>

            <div>
              <h4 className="font-bold text-slate-900 dark:text-slate-100">{exp.title}</h4>
              <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">
                {exp.role} • {exp.period}
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-300">{exp.description}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
