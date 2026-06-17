/**
 * dronmakr UI bootstrap — custom scrollbars (OverlayScrollbars), selects, number steppers.
 */
(function () {
  "use strict";

  var selectInstances = new WeakMap();
  var selectPending = new WeakMap();
  var scrollbarInstances = new WeakMap();
  var scrollbarPending = new WeakMap();

  var scrollbarOptions = {
    scrollbars: {
      theme: "os-theme-dronmakr",
      autoHide: "leave",
      autoHideDelay: 900,
      autoHideSuspend: true,
      visibility: "auto",
      dragScroll: true,
      clickScroll: true,
    },
  };

  function getOverlayScrollbars() {
    var global = window.OverlayScrollbarsGlobal;
    return global && global.OverlayScrollbars;
  }

  function shouldSkipSelect(select) {
    if (!select || select.tagName !== "SELECT") return true;
    if (select.dataset.nativeUi === "true") return true;
    if (select.closest("[data-native-ui]")) return true;
    return false;
  }

  function shouldSkipNumber(input) {
    if (!input || input.tagName !== "INPUT" || input.type !== "number") return true;
    if (input.dataset.nativeUi === "true") return true;
    if (input.dataset.dronUiNumber === "true") return true;
    if (input.closest("[data-native-ui]")) return true;
    if (input.closest(".dron-number-field")) return true;
    if (input.classList.contains("daw-knob-input")) return true;
    if (input.getAttribute("aria-hidden") === "true") return true;
    if (input.tabIndex === -1) return true;
    return false;
  }

  function scrollLike(value) {
    return value === "auto" || value === "scroll" || value === "overlay";
  }

  function shouldSkipScrollbar(el) {
    if (!el || el.nodeType !== 1) return true;
    if (el.tagName === "HTML") return true;
    if (el.dataset.nativeUi === "true") return true;
    if (el.closest("[data-native-ui]")) return true;
    if (el.dataset.dronUiScrollbar === "true") return true;
    if (el.hasAttribute("data-overlayscrollbars")) return true;
    if (el.classList.contains("os-scrollbar")) return true;
    if (el.closest(".os-size-observer, .os-trinsic-observer")) return true;
    return false;
  }

  function isOverflowScrollable(el) {
    if (!el || el.nodeType !== 1) return false;
    var tag = el.tagName;
    if (tag === "HTML" || tag === "SELECT" || tag === "TEXTAREA") return false;

    var style = window.getComputedStyle(el);
    var overflow = style.overflow;
    var overflowX = style.overflowX;
    var overflowY = style.overflowY;

    if (tag === "BODY") {
      if (overflow === "hidden" || overflowY === "hidden") return false;
      return (
        scrollLike(overflowY) ||
        scrollLike(overflow) ||
        el.scrollHeight > el.clientHeight + 1 ||
        el.scrollWidth > el.clientWidth + 1
      );
    }

    return scrollLike(overflowY) || scrollLike(overflowX) || scrollLike(overflow);
  }

  function searchableFor(select) {
    return select.options && select.options.length > 12;
  }

  function bindSelect(select) {
    if (shouldSkipSelect(select)) return;
    if (typeof window.NiceSelect === "undefined") return;
    if (selectInstances.has(select)) {
      selectInstances.get(select).update();
      return;
    }
    select.classList.add("wide");
    var instance = window.NiceSelect.bind(select, {
      searchable: searchableFor(select),
    });
    selectInstances.set(select, instance);
    select.dataset.dronUiBound = "true";
  }

  function unbindSelect(select) {
    var instance = selectInstances.get(select);
    if (!instance) return;
    instance.destroy();
    selectInstances.delete(select);
    delete select.dataset.dronUiBound;
  }

  function refreshSelects(root) {
    var scope = root || document;
    scope.querySelectorAll("select").forEach(function (select) {
      if (shouldSkipSelect(select)) return;
      bindSelect(select);
    });
  }

  function scheduleSelectRefresh(select) {
    if (shouldSkipSelect(select)) return;
    clearTimeout(selectPending.get(select));
    selectPending.set(
      select,
      setTimeout(function () {
        bindSelect(select);
      }, 0)
    );
  }

  function bindScrollbar(el) {
    if (shouldSkipScrollbar(el) || !isOverflowScrollable(el)) return;
    var OverlayScrollbars = getOverlayScrollbars();
    if (!OverlayScrollbars) return;
    if (scrollbarInstances.has(el)) return;

    if (el.tagName === "BODY") {
      document.documentElement.setAttribute("data-overlayscrollbars-initialize", "");
    }
    el.setAttribute("data-overlayscrollbars-initialize", "");

    var instance = OverlayScrollbars(el, scrollbarOptions);
    if (!instance) {
      if (el.tagName === "BODY") {
        document.documentElement.removeAttribute("data-overlayscrollbars-initialize");
      }
      el.removeAttribute("data-overlayscrollbars-initialize");
      return;
    }

    scrollbarInstances.set(el, instance);
    el.dataset.dronUiScrollbar = "true";
  }

  function unbindScrollbar(el) {
    var instance = scrollbarInstances.get(el);
    if (!instance) return;
    instance.destroy();
    scrollbarInstances.delete(el);
    delete el.dataset.dronUiScrollbar;
    el.removeAttribute("data-overlayscrollbars-initialize");
    if (el.tagName === "BODY") {
      document.documentElement.removeAttribute("data-overlayscrollbars-initialize");
    }
  }

  function resolveElementRoot(root) {
    if (root && root.nodeType === 1) return root;
    return document.body;
  }

  function scanScrollbars(root) {
    var scope = resolveElementRoot(root);
    if (!scope) return;
    if (isOverflowScrollable(scope)) bindScrollbar(scope);
    scope.querySelectorAll("*").forEach(function (el) {
      if (isOverflowScrollable(el)) bindScrollbar(el);
    });
  }

  function refreshScrollbars(root) {
    scanScrollbars(root);
  }

  function scheduleScrollbarRefresh(el) {
    if (shouldSkipScrollbar(el)) return;
    clearTimeout(scrollbarPending.get(el));
    scrollbarPending.set(
      el,
      setTimeout(function () {
        if (isOverflowScrollable(el)) bindScrollbar(el);
      }, 0)
    );
  }

  function decimalPlaces(step) {
    var s = String(step);
    var dot = s.indexOf(".");
    return dot === -1 ? 0 : s.length - dot - 1;
  }

  function stepNumber(input, direction) {
    var stepRaw = input.step;
    var step = parseFloat(stepRaw);
    if (!Number.isFinite(step) || stepRaw === "any" || stepRaw === "") step = 1;
    var min = input.min !== "" ? parseFloat(input.min) : null;
    var max = input.max !== "" ? parseFloat(input.max) : null;
    var val = input.value === "" ? 0 : parseFloat(input.value);
    if (!Number.isFinite(val)) val = 0;
    var places = decimalPlaces(step);
    var next = val + direction * step;
    if (places > 0) next = parseFloat(next.toFixed(places));
    if (min != null && Number.isFinite(min)) next = Math.max(min, next);
    if (max != null && Number.isFinite(max)) next = Math.min(max, next);
    input.value = String(next);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function wrapNumberInput(input) {
    if (shouldSkipNumber(input)) return;

    var wrap = document.createElement("div");
    wrap.className = "dron-number-field";
    input.parentNode.insertBefore(wrap, input);
    wrap.appendChild(input);

    var steppers = document.createElement("div");
    steppers.className = "dron-number-steppers";

    var up = document.createElement("button");
    up.type = "button";
    up.className = "dron-number-step dron-number-step-up";
    up.tabIndex = -1;
    up.setAttribute("aria-label", "Increase value");
    up.textContent = "\u25B2";

    var down = document.createElement("button");
    down.type = "button";
    down.className = "dron-number-step dron-number-step-down";
    down.tabIndex = -1;
    down.setAttribute("aria-label", "Decrease value");
    down.textContent = "\u25BC";

    up.addEventListener("click", function (e) {
      e.preventDefault();
      if (input.disabled) return;
      input.focus();
      stepNumber(input, 1);
    });
    down.addEventListener("click", function (e) {
      e.preventDefault();
      if (input.disabled) return;
      input.focus();
      stepNumber(input, -1);
    });

    steppers.appendChild(up);
    steppers.appendChild(down);
    wrap.appendChild(steppers);
    input.dataset.dronUiNumber = "true";
  }

  function refreshNumbers(root) {
    var scope = root || document;
    scope.querySelectorAll('input[type="number"]').forEach(function (input) {
      if (shouldSkipNumber(input)) return;
      wrapNumberInput(input);
    });
  }

  function refresh(root) {
    refreshSelects(root);
    refreshNumbers(root);
    refreshScrollbars(root);
  }

  function walkAddedNode(node, handler) {
    if (node.nodeType !== 1) return;
    handler(node);
    if (node.querySelectorAll) {
      node.querySelectorAll("*").forEach(handler);
    }
  }

  function initObserver() {
    if (!document.body) return;
    var observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        if (mutation.type === "attributes") {
          var target = mutation.target;
          if (target.tagName === "SELECT") scheduleSelectRefresh(target);
          if (target.nodeType === 1) scheduleScrollbarRefresh(target);
          return;
        }
        if (mutation.type !== "childList") return;

        mutation.addedNodes.forEach(function (node) {
          walkAddedNode(node, function (el) {
            if (el.tagName === "SELECT") scheduleSelectRefresh(el);
            if (el.tagName === "INPUT" && el.type === "number") wrapNumberInput(el);
            scheduleScrollbarRefresh(el);
          });
        });

        mutation.removedNodes.forEach(function (node) {
          walkAddedNode(node, function (el) {
            if (el.tagName === "SELECT") unbindSelect(el);
            unbindScrollbar(el);
          });
        });

        if (mutation.target.tagName === "SELECT") {
          scheduleSelectRefresh(mutation.target);
        }
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class", "disabled", "style"],
    });
  }

  function init() {
    var global = window.OverlayScrollbarsGlobal;
    if (global && global.OverlayScrollbars) {
      global.OverlayScrollbars.plugin([
        global.ScrollbarsHidingPlugin,
        global.SizeObserverPlugin,
        global.ClickScrollPlugin,
      ]);
    }
    refresh(document);
    initObserver();
    window.addEventListener(
      "resize",
      (function () {
        var timer;
        return function () {
          clearTimeout(timer);
          timer = setTimeout(function () {
            refreshScrollbars(document.body);
          }, 150);
        };
      })(),
      { passive: true }
    );
  }

  window.dronmakrUi = {
    refresh: refresh,
    refreshSelects: refreshSelects,
    refreshNumbers: refreshNumbers,
    refreshScrollbars: refreshScrollbars,
    bindSelect: bindSelect,
    unbindSelect: unbindSelect,
    bindScrollbar: bindScrollbar,
    unbindScrollbar: unbindScrollbar,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
