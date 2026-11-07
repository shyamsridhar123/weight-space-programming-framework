(() => {
  'use strict';

  const views = ['weights', 'futures'];
  const tabs = [...document.querySelectorAll('.lw-page-tabs [role="tab"]')];

  function showView(name, { focus = false, updateHash = false } = {}) {
    const active = views.includes(name) ? name : 'weights';
    for (const view of views) document.getElementById(view).hidden = view !== active;
    for (const tab of tabs) {
      const selected = tab.dataset.labView === active;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focus) tab.focus();
    }
    if (updateHash && location.hash !== `#${active}`) {
      history.replaceState(null, '', `#${active}`);
    }
  }

  for (const [index, tab] of tabs.entries()) {
    tab.addEventListener('click', () => showView(tab.dataset.labView, { updateHash: true }));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault();
      showView(tabs[next].dataset.labView, { focus: true, updateHash: true });
    });
  }

  for (const link of document.querySelectorAll('[data-lab-go]')) {
    link.addEventListener('click', () => {
      showView(link.dataset.labGo, { updateHash: true });
      document.getElementById(link.dataset.labGo).focus();
    });
  }

  window.addEventListener('hashchange', () => showView(location.hash.slice(1)));
  showView(location.hash.slice(1));
})();
