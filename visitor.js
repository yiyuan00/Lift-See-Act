(() => {
  const card = document.getElementById("visitor-welcome");
  if (!card || !card.dataset.counterUrl) return;

  const numberFormat = new Intl.NumberFormat("en-US");
  const title = document.getElementById("visitor-welcome-title");
  const message = document.getElementById("visitor-welcome-message");
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

  const privacyNotice = document.getElementById("visitor-privacy");
  privacyNotice.hidden = false;
  if (navigator.globalPrivacyControl || navigator.doNotTrack === "1" || window.doNotTrack === "1") {
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
  fetch(`${card.dataset.counterUrl.replace(/\/$/, "")}/visit`, {
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
    document.getElementById("visitor-total").textContent = numberFormat.format(data.visitors);
    document.getElementById("visitor-views").textContent = numberFormat.format(data.views);
    document.getElementById("visitor-stats").hidden = false;
    showWelcome(data.isNew ? "Welcome to LSA!" : "Welcome back to LSA!", data.visitorNumber);
    if (!remembered) message.textContent += " Your browser cannot save this number.";
  }).catch(() => {
    // A failed request never invents a visitor number or a local "global" count.
    showWelcome("Welcome to LSA!");
  }).finally(() => window.clearTimeout(timeout));
})();
