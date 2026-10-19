// Keep the theme's back-to-top control above the footer, as on the reference site.
const backToTop = document.getElementById("back-to-top");
const siteFooter = document.querySelector("footer.sticky-bottom");

if (backToTop && siteFooter) {
  const positionBackToTop = () => {
    const footerOverlap = window.innerHeight - siteFooter.getBoundingClientRect().top;
    backToTop.style.bottom = `${Math.max(0, footerOverlap) + 15}px`;
  };

  backToTop.setAttribute("role", "button");
  backToTop.setAttribute("aria-label", "Back to top");
  backToTop.setAttribute("tabindex", "0");
  backToTop.querySelector("svg")?.setAttribute("aria-hidden", "true");
  backToTop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      backToTop.click();
    }
  });

  window.addEventListener("scroll", positionBackToTop, { passive: true });
  window.addEventListener("resize", positionBackToTop);
  positionBackToTop();
}
