(function () {
  const header = document.querySelector(".site-header");
  const button = document.querySelector(".menu-btn");
  const nav = document.querySelector(".primary-nav");

  if (header && button && nav) {
    const setOpen = (open) => {
      header.classList.toggle("is-open", open);
      button.setAttribute("aria-expanded", open ? "true" : "false");
      button.textContent = open ? "Close" : "Menu";
    };

    button.addEventListener("click", () => {
      setOpen(!header.classList.contains("is-open"));
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") setOpen(false);
    });
  }

  const copyBtn = document.querySelector("[data-copy]");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const target = document.querySelector(copyBtn.getAttribute("data-copy"));
      if (!target) return;
      const text = target.textContent || "";
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = "Copied";
        copyBtn.setAttribute("aria-live", "polite");
        window.setTimeout(() => {
          copyBtn.textContent = "Copy BibTeX";
        }, 2000);
      } catch (_err) {
        copyBtn.textContent = "Select the text";
      }
    });
  }
})();
