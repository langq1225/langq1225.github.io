import { useState } from 'react';
import profileImage from '../assets/images/me.jpg';

export default function Hero() {
  const [isContactOpen, setIsContactOpen] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);

  const toggleContact = () => {
    setIsContactOpen(!isContactOpen);
  };

  return (
    <section
      id="about"
      className="opacity-0 animate-fade-in-up scroll-mt-14 md:scroll-mt-20"
    >
      <div className="bg-white dark:bg-slate-800/80 rounded-3xl p-8 md:p-12 shadow-xl border border-white/20 dark:border-slate-700 relative overflow-hidden">
        {/* Decorative Background Blob */}
        <div className="absolute top-0 right-0 -mt-20 -mr-20 w-64 h-64 bg-primary-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 dark:opacity-20 animate-pulse-slow"></div>
        <div
          className="absolute bottom-0 left-0 -mb-20 -ml-20 w-64 h-64 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-10 dark:opacity-20 animate-pulse-slow"
          style={{ animationDelay: '1s' }}
        ></div>

        <div className="relative z-10 flex flex-col md:flex-row gap-10 items-start">
          {/* Profile Image with Ring */}
          <div className="relative group mx-auto md:mx-0">
            <div className="absolute -inset-1 bg-gradient-to-r from-primary-600 to-indigo-600 rounded-full blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
            <div className="relative w-40 h-40 md:w-48 md:h-48 rounded-full overflow-hidden border-4 border-white dark:border-slate-800 shadow-lg">
              {!imageLoaded && (
                <div className="absolute inset-0 bg-slate-200 dark:bg-slate-700 animate-pulse" />
              )}
              <img
                src={profileImage}
                alt="Langqing Cui"
                loading="eager"
                fetchpriority="high"
                className={`w-full h-full object-cover transform group-hover:scale-105 transition duration-500 ${
                  imageLoaded ? 'opacity-100' : 'opacity-0'
                }`}
                onLoad={() => setImageLoaded(true)}
                onError={(e) => {
                  e.target.src = 'https://ui-avatars.com/api/?name=Langqing+Cui&background=0D8ABC&color=fff&size=256';
                  setImageLoaded(true);
                }}
              />
            </div>
          </div>

          {/* Bio */}
          <div className="flex-1 text-center md:text-left space-y-4 w-full">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-slate-100 tracking-tight mb-2">
                Langqing Cui
                <span className="text-2xl font-normal text-slate-500 block mt-2 md:mt-0 md:inline md:ml-2">
                  崔朗清
                </span>
              </h1>
              <p className="text-lg text-primary-600 dark:text-primary-400 font-medium">
                M.S. Student in EE @ KAIST
              </p>
            </div>

            <p className="text-slate-600 dark:text-slate-300 leading-relaxed max-w-2xl">
              I am a first-year M.S. student in the{' '}
              <a
                href="https://ee.kaist.ac.kr/en/"
                className="text-primary-600 hover:underline decoration-2 underline-offset-2"
              >
                School of Electrical Engineering
              </a>{' '}
              at KAIST (Korea Advanced Institute of Science and Technology), advised by Prof.{' '}
              <a
                href="https://insuhan.github.io/"
                className="text-primary-600 hover:underline decoration-2 underline-offset-2"
              >
                Insu Han
              </a>
              . Previously, I obtained my B.Eng. in Computer Science and Technology at{' '}
              <a
                href="https://english.bjut.edu.cn/"
                className="text-primary-600 hover:underline decoration-2 underline-offset-2"
              >
                Beijing University of Technology
              </a>
              . <br />
              <br />
              I have a broad research interest on <b>Efficient Foundation Models</b>. Currently, I
              focus on <b>KV Cache Management</b> and <b>Diffusion Models</b>.
            </p>

            {/* Buttons */}
            <div className="flex flex-col md:items-start items-center pt-2">
              <div className="flex flex-wrap justify-center md:justify-start gap-3">
                {/* Contact Button (Toggle) */}
                <button
                  onClick={toggleContact}
                  className="px-5 py-2.5 bg-slate-900 text-white dark:bg-slate-200 dark:text-slate-900 rounded-full font-medium hover:bg-slate-700 dark:hover:bg-white transition-all hover:scale-105 active:scale-95 shadow-md flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-slate-900 dark:focus:ring-slate-400"
                >
                  <i className="fas fa-address-card"></i> Contact Me
                </button>

                {/* LinkedIn Button */}
                <a
                  href="https://www.linkedin.com/in/langqing-cui/"
                  className="px-5 py-2.5 bg-[#ffffff] dark:bg-slate-800 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-600 rounded-full font-medium hover:bg-gray-50 dark:hover:bg-slate-700 transition-all hover:scale-105 active:scale-95 shadow-sm flex items-center gap-2"
                >
                  <i className="fab fa-linkedin"></i> LinkedIn
                </a>
              </div>

              {/* Collapsible Contact Info */}
              <div
                className={`w-full max-w-md overflow-hidden transition-all duration-500 ease-in-out ${
                  isContactOpen ? 'max-h-96 opacity-100 mt-4' : 'max-h-0 opacity-0 mt-0'
                }`}
              >
                {/* Inner Card */}
                <div className="p-4 bg-slate-50 dark:bg-slate-800/60 rounded-xl border border-slate-200 dark:border-slate-600 text-left">
                  <div className="space-y-3">
                    {/* Email Section */}
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center text-primary-600 dark:text-primary-400 shrink-0">
                        <i className="fas fa-envelope"></i>
                      </div>
                      <div className="flex-1">
                        <p className="text-xs text-slate-500 dark:text-slate-400 font-semibold uppercase">
                          Email
                        </p>
                        <p className="text-slate-800 dark:text-slate-200 text-sm break-all font-medium">
                          langqing [dot] cui [at] outlook [dot] com
                        </p>
                      </div>
                    </div>

                    {/* Address Section */}
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center text-primary-600 dark:text-primary-400 shrink-0">
                        <i className="fas fa-map-marker-alt"></i>
                      </div>
                      <div className="flex-1">
                        <p className="text-xs text-slate-500 dark:text-slate-400 font-semibold uppercase">
                          Address
                        </p>
                        <p className="text-slate-800 dark:text-slate-200 text-sm font-medium">
                          N1 917, 291 Daehak-ro, Daejeon, South Korea
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
