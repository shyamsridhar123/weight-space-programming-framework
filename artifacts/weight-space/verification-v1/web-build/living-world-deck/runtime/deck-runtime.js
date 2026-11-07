(() => {
  "use strict";
  if (new URLSearchParams(location.search).get("mode") === "audience") {
    document.title = "Crown & Charter — Audience";
  }
  if (location.protocol === "file:") {
    document.querySelector("hyperframes-slideshow").hidden = true;
    document.getElementById("local-server-help").hidden = false;
    return;
  }

  // Configure the local composition before its first navigation. Its export
  // controls download JSON, and its source/live-game links open separate tabs.
  for (const player of document.querySelectorAll("hyperframes-player[data-composition-src]")) {
    const frame = player.iframeElement;
    if (!frame) throw new Error("The bundled HyperFrames player did not load.");
    frame.sandbox.add("allow-downloads", "allow-popups", "allow-popups-to-escape-sandbox");
    player.setAttribute("src", player.dataset.compositionSrc);
  }
})();
