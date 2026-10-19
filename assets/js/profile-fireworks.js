(() => {
  const trigger = document.querySelector(".profile-fireworks-trigger");
  const portrait = document.querySelector(".profile-portrait img");
  if (!trigger || !portrait) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const colors = ["#578cff", "#37b9aa", "#efa63d", "#d982b5"];
  const crestUrls = [trigger.dataset.crestArsenal, trigger.dataset.crestGuoan].filter(Boolean);
  const crestChance = 0.35;
  let crests = [];
  let preloadStarted = false;
  let canvas;
  let context;
  let bounds;
  let particles = [];
  let frame = 0;
  let lastBurst = -Infinity;

  // Original crests: Wikimedia's Arsenal_FC.svg and Beijing_Guoan_F.C.svg.
  // Load the two small local SVGs once, only when someone approaches the egg.
  const preloadCrests = () => {
    if (preloadStarted || reducedMotion.matches) return;
    preloadStarted = true;
    Promise.all(
      crestUrls.map(async (src) => {
        const image = new Image();
        image.decoding = "async";
        image.src = src;
        try {
          await image.decode();
          return image;
        } catch {
          return null;
        }
      })
    ).then((images) => {
      crests = images.filter(Boolean);
    });
  };

  const clearFireworks = () => {
    cancelAnimationFrame(frame);
    frame = 0;
    particles = [];
    canvas?.remove();
    canvas = null;
    context = null;
  };

  const draw = (now) => {
    context.clearRect(0, 0, bounds.width, bounds.height);
    particles = particles.filter((particle) => now - particle.start < particle.duration);

    for (const particle of particles) {
      const age = now - particle.start;
      if (age < 0) continue;
      const progress = age / particle.duration;
      const seconds = age / 1000;
      const distance = (particle.speed * (1 - Math.exp(-2.4 * seconds))) / 2.4;
      const tailSeconds = Math.max(0, seconds - 0.045);
      const tailDistance = (particle.speed * (1 - Math.exp(-2.4 * tailSeconds))) / 2.4;
      const x = particle.x + Math.cos(particle.angle) * distance;
      const y = particle.y + Math.sin(particle.angle) * distance + 28 * seconds * seconds;

      context.globalAlpha = Math.pow(1 - progress, 1.4);
      if (particle.image) {
        const height = particle.size * (0.8 + 0.2 * Math.min(progress * 6, 1));
        const width = (height * particle.image.naturalWidth) / particle.image.naturalHeight;
        context.save();
        context.translate(x, y);
        context.rotate(particle.rotation * progress);
        context.globalAlpha = Math.min(progress * 8, 1) * Math.pow(1 - progress, 0.6);
        context.drawImage(particle.image, -width / 2, -height / 2, width, height);
        context.restore();
        continue;
      }

      context.strokeStyle = particle.color;
      context.fillStyle = particle.color;
      context.lineWidth = particle.size;
      context.lineCap = "round";
      context.beginPath();
      context.moveTo(
        particle.x + Math.cos(particle.angle) * tailDistance,
        particle.y + Math.sin(particle.angle) * tailDistance + 28 * tailSeconds * tailSeconds
      );
      context.lineTo(x, y);
      context.stroke();

      if (particle.star) {
        const sparkle = 2.5 * (1 - progress);
        context.beginPath();
        context.moveTo(x - sparkle, y);
        context.lineTo(x + sparkle, y);
        context.moveTo(x, y - sparkle);
        context.lineTo(x, y + sparkle);
        context.stroke();
      } else {
        context.beginPath();
        context.arc(x, y, particle.size / 2, 0, Math.PI * 2);
        context.fill();
      }
    }

    if (particles.length) frame = requestAnimationFrame(draw);
    else clearFireworks();
  };

  const celebrate = () => {
    if (reducedMotion.matches) return;
    preloadCrests();

    const now = performance.now();
    if (now - lastBurst < 160) return;
    lastBurst = now;
    const rect = portrait.getBoundingClientRect();

    if (!canvas) {
      const left = Math.max(0, rect.left - 100);
      const top = Math.max(0, rect.top - 100);
      bounds = {
        left,
        top,
        width: Math.min(window.innerWidth, rect.right + 100) - left,
        height: Math.min(window.innerHeight, rect.bottom + 100) - top,
      };
      canvas = document.createElement("canvas");
      canvas.className = "profile-fireworks-canvas";
      canvas.setAttribute("aria-hidden", "true");
      const scale = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.ceil(bounds.width * scale);
      canvas.height = Math.ceil(bounds.height * scale);
      Object.assign(canvas.style, {
        left: `${left}px`,
        top: `${top}px`,
        width: `${bounds.width}px`,
        height: `${bounds.height}px`,
      });
      context = canvas.getContext("2d");
      if (!context) {
        clearFireworks();
        return;
      }
      context.scale(scale, scale);
      document.body.append(canvas);
    }

    const bursts = [
      [rect.left - 8, rect.top + rect.height * 0.3],
      [rect.right + 8, rect.top + rect.height * 0.48],
      [rect.left + rect.width * 0.64, rect.top - 8],
    ];
    bursts.forEach(([x, y], burst) => {
      for (let index = 0; index < 24; index++) {
        particles.push({
          x: x - bounds.left,
          y: y - bounds.top,
          angle: (index / 24) * Math.PI * 2 + Math.random() * 0.12,
          speed: 110 + Math.random() * 85,
          color: colors[(index + burst) % colors.length],
          size: 1.2 + Math.random() * 0.8,
          star: index % 6 === 0,
          start: now + burst * 90,
          duration: 650 + Math.random() * 350,
        });
      }
    });

    // At most one rare crest per click, and two on screen during rapid clicking.
    if (crests.length && Math.random() < crestChance && particles.filter((particle) => particle.image).length < 2) {
      const burst = Math.floor(Math.random() * bursts.length);
      const [x, y] = bursts[burst];
      particles.push({
        x: x - bounds.left,
        y: y - bounds.top,
        angle: Math.atan2(y - rect.top - rect.height / 2, x - rect.left - rect.width / 2) - 0.15,
        speed: 60 + Math.random() * 30,
        image: crests[Math.floor(Math.random() * crests.length)],
        rotation: (Math.random() - 0.5) * 0.7,
        size: 20 + Math.random() * 4,
        start: now + burst * 90,
        duration: 950,
      });
    }
    const crestParticles = particles.filter((particle) => particle.image);
    particles = particles
      .filter((particle) => !particle.image)
      .slice(-(216 - crestParticles.length))
      .concat(crestParticles);
    if (!frame) frame = requestAnimationFrame(draw);
  };

  trigger.addEventListener("pointerenter", preloadCrests, { once: true });
  trigger.addEventListener("pointerdown", preloadCrests, { once: true });
  trigger.addEventListener("focus", preloadCrests, { once: true });
  trigger.addEventListener("click", celebrate);
  window.addEventListener("scroll", clearFireworks, { passive: true });
  window.addEventListener("resize", clearFireworks);
  reducedMotion.addEventListener("change", clearFireworks);
})();
