(() => {
  const card = document.getElementById("visitor-welcome");
  if (!card || !card.dataset.counterUrl) return;

  const numberFormat = new Intl.NumberFormat("en-US");
  const title = document.getElementById("visitor-welcome-title");
  const message = document.getElementById("visitor-welcome-message");
  const calendar = document.getElementById("visitor-calendar");
  const totalNumber = document.getElementById("visitor-total");
  const flipLeaf = calendar?.querySelector(".visitor-calendar-flip");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const counterUrl = card.dataset.counterUrl.replace(/\/$/, "");
  let lastTotal = -1;
  let flipTimer;
  const updateTotal = (total) => {
    if (!calendar || !Number.isSafeInteger(total) || total < 0 || total < lastTotal) return;
    calendar.setAttribute("aria-busy", "false");
    calendar.setAttribute("aria-label", `${numberFormat.format(total)} total browser visitors`);
    calendar.title = `Total browser visitors · updated ${new Date().toLocaleTimeString("en-US")}`;
    if (total === lastTotal) return;
    const previous = totalNumber.textContent;
    lastTotal = total;
    totalNumber.textContent = numberFormat.format(total);
    calendar.dataset.digits = String(Math.min(6, String(total).length));
    window.clearTimeout(flipTimer);
    calendar.classList.remove("is-flipping");
    if (!reduceMotion && flipLeaf) {
      flipLeaf.textContent = previous;
      // Reading layout restarts the turn only when the real total changes.
      void calendar.offsetWidth;
      calendar.classList.add("is-flipping");
      flipTimer = window.setTimeout(() => calendar.classList.remove("is-flipping"), 750);
    }
  };
  const readTotal = async () => {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 6000);
    try {
      const response = await fetch(`${counterUrl}/stats`, { credentials: "omit", referrerPolicy: "no-referrer", cache: "no-store", signal: controller.signal });
      if (!response.ok) throw new Error("Counter unavailable");
      const data = await response.json();
      if (!Number.isSafeInteger(data.visitors) || data.visitors < 0) throw new Error("Invalid count");
      updateTotal(data.visitors);
    } catch {
      if (calendar && lastTotal < 0) {
        calendar.setAttribute("aria-busy", "false");
        calendar.setAttribute("aria-label", "Total visitors temporarily unavailable");
        calendar.title = "Visitor count temporarily unavailable";
      }
    } finally { window.clearTimeout(timeout); }
  };
  // Aggregate-only reads do not assign a visitor number or add a page view.
  window.setInterval(() => { if (!document.hidden) void readTotal(); }, 60000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) void readTotal(); });
  let dismissTimer;
  const dismiss = () => { card.hidden = true; window.clearTimeout(dismissTimer); };
  const scheduleDismiss = () => {
    window.clearTimeout(dismissTimer);
    dismissTimer = window.setTimeout(dismiss, 9000);
  };
  const showWelcome = (text, visitorNumber) => {
    title.textContent = text;
    message.textContent = visitorNumber ? `You're visitor #${numberFormat.format(visitorNumber)}. Thanks for stopping by!` : "Thanks for stopping by!";
    card.hidden = false;
    scheduleDismiss();
  };
  card.querySelector(".visitor-welcome-close").addEventListener("click", dismiss);
  card.addEventListener("mouseenter", () => window.clearTimeout(dismissTimer));
  card.addEventListener("mouseleave", scheduleDismiss);
  card.addEventListener("focusin", () => window.clearTimeout(dismissTimer));
  card.addEventListener("focusout", scheduleDismiss);
  card.addEventListener("keydown", (event) => { if (event.key === "Escape") dismiss(); });

  if (navigator.globalPrivacyControl || navigator.doNotTrack === "1" || window.doNotTrack === "1") {
    void readTotal();
    showWelcome("Welcome to LSA!");
    return;
  }

  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  let browserId;
  let remembered = false;
  try {
    browserId = localStorage.getItem("lsa-browser-id-v1");
    if (!uuidPattern.test(browserId || "")) {
      browserId = crypto.randomUUID();
      localStorage.setItem("lsa-browser-id-v1", browserId);
    }
    remembered = true;
  } catch {
    browserId = crypto.randomUUID();
  }
  const visitId = crypto.randomUUID();
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 6000);
  fetch(`${counterUrl}/visit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ browserId, visitId }),
    credentials: "omit",
    referrerPolicy: "no-referrer",
    cache: "no-store",
    signal: controller.signal
  }).then(async (response) => {
    if (!response.ok) throw new Error("Counter unavailable");
    const data = await response.json();
    if (![data.visitorNumber, data.visitors, data.views].every(value => Number.isSafeInteger(value) && value > 0)
        || data.visitorNumber > data.visitors || data.visitors > data.views || typeof data.isNew !== "boolean") {
      throw new Error("Invalid counter response");
    }
    updateTotal(data.visitors);
    showWelcome(data.isNew ? "Welcome to LSA!" : "Welcome back to LSA!", data.visitorNumber);
    if (!remembered) message.textContent += " Your browser cannot save this number.";
  }).catch(() => {
    // A failed request never invents a visitor number or a local "global" count.
    showWelcome("Welcome to LSA!");
    void readTotal();
  }).finally(() => window.clearTimeout(timeout));
})();
