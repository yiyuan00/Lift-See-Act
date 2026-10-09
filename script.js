document.addEventListener("DOMContentLoaded", () => {
  const revealItems = [...document.querySelectorAll("[data-reveal]")];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduceMotion || !("IntersectionObserver" in window)) {
    revealItems.forEach((item) => item.classList.add("is-visible"));
  } else {
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    revealItems.forEach((item) => revealObserver.observe(item));
  }

  const lazyVideos = [...document.querySelectorAll("video.lazy")];

  const loadVideo = (video) => {
    [...video.children].forEach((source) => {
      if (source.tagName === "SOURCE" && source.dataset.src) {
        source.src = source.dataset.src;
      }
    });
    video.load();
    video.classList.remove("lazy");
    video.play().catch(() => {});
  };

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          loadVideo(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { rootMargin: "300px 0px" });

    lazyVideos.forEach((video) => observer.observe(video));
  } else {
    lazyVideos.forEach(loadVideo);
  }

  document.querySelectorAll(".demo-item video").forEach((video) => {
    video.addEventListener("click", () => {
      if (video.paused) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  });

  const copyButton = document.getElementById("copy-citation");
  const bibtex = document.getElementById("bibtex");
  const backToTop = document.querySelector(".back-to-top");

  const updateBackToTop = () => {
    backToTop?.classList.toggle("is-visible", window.scrollY > 720);
  };
  window.addEventListener("scroll", updateBackToTop, { passive: true });
  updateBackToTop();

  copyButton?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      copyButton.querySelector("span").textContent = "Copied";
      window.setTimeout(() => { copyButton.querySelector("span").textContent = "Copy"; }, 1600);
    } catch {
      copyButton.querySelector("span").textContent = "Select text to copy";
    }
  });
});
