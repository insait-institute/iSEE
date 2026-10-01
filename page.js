// iSEE project page: lazy play/pause of the clips, the hidden-frame timeline, comparison tabs, BibTeX copy.
// Without this script the page still works: every clip has controls and a poster, and every comparison is shown.
(() => {
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const live = new Set();

  // A clip plays only while it is on screen (the videos are preload="none", so an off-screen clip costs nothing).
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target.querySelector("video");
      if (e.isIntersecting) { live.add(e.target); if (!still) v.play().catch(() => {}); }
      else { live.delete(e.target); v.pause(); }
    }
  }, { rootMargin: "120px 0px" });

  for (const c of document.querySelectorAll(".clip[data-n]")) {
    const v = c.querySelector("video"), tl = c.querySelector(".tl"), ph = c.querySelector(".ph");
    const n = +c.dataset.n, fps = +c.dataset.fps;
    if (!still) v.removeAttribute("controls");
    // The playhead marks the frame on screen; grey spans are the frames on which the target is fully hidden.
    c.draw = () => {
      const f = Math.min(n - 1, Math.floor(v.currentTime * fps));
      ph.style.left = ((f + 0.5) / n) * 100 + "%";
    };
    tl.addEventListener("click", (ev) => {
      const r = tl.getBoundingClientRect();
      const t = ((ev.clientX - r.left) / r.width) * n / fps;
      if (Number.isFinite(t)) { v.currentTime = t; c.draw(); }
    });
    io.observe(c);
  }
  const tick = () => { for (const c of live) c.draw(); requestAnimationFrame(tick); };
  requestAnimationFrame(tick);

  // Comparison tabs: one clip at a time.
  for (const box of document.querySelectorAll("[data-tabs]")) {
    const panes = [...box.querySelectorAll(":scope > .stack > figure")];
    const bar = document.createElement("div");
    bar.className = "tabbar"; bar.setAttribute("role", "tablist");
    const show = (i) => panes.forEach((p, j) => {
      p.hidden = i !== j;
      bar.children[j].setAttribute("aria-selected", String(i === j));
    });
    panes.forEach((p, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "tab"); b.textContent = p.dataset.tab;
      b.addEventListener("click", () => show(i));
      bar.append(b);
    });
    box.prepend(bar);
    show(0);
  }

  // YouTube: a thumbnail link until clicked; only then does the player (youtube-nocookie) load.
  for (const a of document.querySelectorAll("a.yt")) {
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const f = document.createElement("iframe");
      f.className = "yt";
      f.src = `https://www.youtube-nocookie.com/embed/${a.dataset.id}?autoplay=1&rel=0&playsinline=1&start=${a.dataset.start}`;
      f.title = a.getAttribute("aria-label");
      f.allow = "autoplay; encrypted-media; picture-in-picture; fullscreen";
      f.allowFullscreen = true;
      a.replaceWith(f);
    });
  }

  const copy = document.getElementById("copy-bib");
  if (copy) copy.addEventListener("click", () => {
    const text = document.getElementById("bibtex").innerText;
    const done = () => { copy.textContent = "Copied"; setTimeout(() => (copy.textContent = "Copy"), 1500); };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => {});
    else {
      const r = document.createRange(); r.selectNodeContents(document.getElementById("bibtex"));
      const s = getSelection(); s.removeAllRanges(); s.addRange(r); document.execCommand("copy"); done();
    }
  });
})();
