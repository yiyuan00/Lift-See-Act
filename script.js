document.addEventListener("DOMContentLoaded", () => {
  const revealItems = [...document.querySelectorAll("[data-reveal]")];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const reviewerTrophy = document.querySelector(".reviewer-trophy");
  const confettiLayer = document.querySelector(".reviewer-confetti");
  let celebrationActive = false;
  reviewerTrophy?.addEventListener("click", () => {
    reviewerTrophy.classList.add("is-stopped");
    reviewerTrophy.setAttribute("aria-pressed", "true");
    reviewerTrophy.title = "Congrats! Click for more confetti.";
    if (!confettiLayer || celebrationActive) return;
    celebrationActive = true;
    const colors = ["#ef767a", "#3e9fbd", "#9a6bb8", "#e8bb50", "#86bd91"];
    const pieceCount = reduceMotion ? 12 : 44;
    confettiLayer.style.setProperty("--fall-height", `${confettiLayer.clientHeight + 60}px`);
    const pieces = document.createDocumentFragment();
    for (let index = 0; index < pieceCount; index += 1) {
      const piece = document.createElement("span");
      piece.className = "reviewer-confetti-piece";
      if (index % 8 === 0) {
        piece.classList.add("reviewer-confetti-flower");
        piece.textContent = "🌸";
      }
      piece.style.left = `${((index + Math.random()) / pieceCount) * 100}%`;
      piece.style.backgroundColor = colors[index % colors.length];
      piece.style.setProperty("--fall-duration", `${2.5 + Math.random() * 1.1}s`);
      piece.style.setProperty("--fall-delay", `${Math.random() * .7}s`);
      piece.style.setProperty("--fall-drift", `${Math.random() * 100 - 50}px`);
      piece.style.setProperty("--fall-spin", `${Math.random() * 720 - 360}deg`);
      piece.addEventListener("animationend", () => piece.remove(), { once: true });
      pieces.appendChild(piece);
    }
    confettiLayer.replaceChildren(pieces);
    window.setTimeout(() => {
      confettiLayer.replaceChildren();
      celebrationActive = false;
    }, reduceMotion ? 1000 : 4500);
  });

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
