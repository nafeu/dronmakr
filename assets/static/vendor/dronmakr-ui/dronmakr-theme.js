/**
 * dronmakr day/night theme — boot early, sync toggles, persist to localStorage + settings.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "dronmakr_ui_theme";
  var VALID = { night: true, day: true };

  function normalizeTheme(value) {
    return value === "day" ? "day" : "night";
  }

  function readStoredTheme() {
    try {
      var stored = localStorage.getItem(STORAGE_KEY);
      if (stored === "day" || stored === "night") return stored;
    } catch (e) {}
    return "night";
  }

  function applyTheme(theme) {
    var next = normalizeTheme(theme);
    document.documentElement.setAttribute("data-theme", next);
    return next;
  }

  function persistTheme(theme, syncSettings) {
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch (e) {}

    if (syncSettings === false) return;

    fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ UI_THEME: theme }),
    }).catch(function () {});
  }

  function updateToggleUi(theme) {
    var isDay = theme === "day";
    var toolbarBtn = document.getElementById("toolbarThemeToggle");
    if (toolbarBtn) {
      toolbarBtn.title = isDay ? "Switch to night mode" : "Switch to day mode";
      toolbarBtn.setAttribute("aria-label", toolbarBtn.title);
      toolbarBtn.innerHTML =
        '<i class="fa-solid ' +
        (isDay ? "fa-moon" : "fa-sun") +
        '" aria-hidden="true"></i>';
    }

    document.querySelectorAll("[data-ui-theme-option]").forEach(function (btn) {
      var option = btn.getAttribute("data-ui-theme-option");
      var active = option === theme;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  function setTheme(theme, options) {
    var opts = options || {};
    var next = applyTheme(normalizeTheme(theme));
    updateToggleUi(next);
    if (opts.persist !== false) persistTheme(next, opts.syncSettings !== false);
    try {
      document.documentElement.dispatchEvent(
        new CustomEvent("dronmakr-themechange", { detail: { theme: next } })
      );
    } catch (e) {}
    return next;
  }

  function toggleTheme() {
    var current = normalizeTheme(document.documentElement.getAttribute("data-theme"));
    return setTheme(current === "day" ? "night" : "day");
  }

  function bindControls() {
    var toolbarBtn = document.getElementById("toolbarThemeToggle");
    if (toolbarBtn && toolbarBtn.dataset.themeBound !== "true") {
      toolbarBtn.dataset.themeBound = "true";
      toolbarBtn.addEventListener("click", function () {
        toggleTheme();
      });
    }

    document.querySelectorAll("[data-ui-theme-option]").forEach(function (btn) {
      if (btn.dataset.themeBound === "true") return;
      btn.dataset.themeBound = "true";
      btn.addEventListener("click", function () {
        var option = btn.getAttribute("data-ui-theme-option");
        if (VALID[option]) setTheme(option);
      });
    });
  }

  function syncThemeFromSettings() {
    fetch("/api/settings")
      .then(function (res) {
        return res.ok ? res.json() : null;
      })
      .then(function (data) {
        if (!data || typeof data.UI_THEME !== "string") {
          updateToggleUi(readStoredTheme());
          return;
        }
        setTheme(normalizeTheme(data.UI_THEME), { syncSettings: false });
      })
      .catch(function () {
        updateToggleUi(readStoredTheme());
      });
  }

  function readColor(name, fallback) {
    try {
      var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      if (value) return value;
    } catch (e) {}
    return fallback || "";
  }

  window.dronmakrTheme = {
    getTheme: function () {
      return normalizeTheme(document.documentElement.getAttribute("data-theme"));
    },
    setTheme: setTheme,
    toggleTheme: toggleTheme,
    readColor: readColor,
  };

  applyTheme(readStoredTheme());

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      bindControls();
      updateToggleUi(readStoredTheme());
      syncThemeFromSettings();
    });
  } else {
    bindControls();
    updateToggleUi(readStoredTheme());
    syncThemeFromSettings();
  }
})();
