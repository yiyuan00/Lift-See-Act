document.addEventListener("DOMContentLoaded", () => {
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

  document.querySelectorAll("video").forEach((video) => {
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

  copyButton?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      copyButton.textContent = "Copied";
      window.setTimeout(() => { copyButton.textContent = "Copy"; }, 1600);
    } catch {
      copyButton.textContent = "Select text to copy";
    }
  });
});
