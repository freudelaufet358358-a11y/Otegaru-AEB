// 画面の入口: 「AEB 合成」と「Leica M10 の色」のタブの切り替え、ドロップされたファイルの振り分け、使い方の表示。

import './style.css';
import { initAeb } from './aeb';
import { ensureCrossOriginIsolation } from './coi';
import { initLeica } from './leica';
import { $ } from './ui';

type TabId = 'aeb' | 'leica';
const TAB_IDS: TabId[] = ['aeb', 'leica'];

let active: TabId = 'aeb';

const leica = initLeica();
const aeb = initAeb({
  isActive: () => active === 'aeb',
  openInLeica: (job) => {
    selectTab('leica');
    leica.openMerged(job);
  },
});
const tabs: Record<TabId, { addFiles: (files: File[]) => void }> = { aeb, leica };

function tabFromHash(): TabId {
  return location.hash === '#leica' ? 'leica' : 'aeb';
}

/** タブを切り替える。URL にも #leica として残す（再読み込みやブックマークで同じタブが開くように） */
function selectTab(id: TabId): void {
  active = id;
  for (const t of TAB_IDS) {
    const on = t === id;
    const tab = $(`tab-${t}`);
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
    $(`panel-${t}`).hidden = !on;
  }
  const hash = id === 'aeb' ? '' : `#${id}`;
  if (location.hash !== hash) history.replaceState(null, '', hash || location.pathname + location.search);
}

function bindTabs(): void {
  for (const t of TAB_IDS) $(`tab-${t}`).onclick = () => selectTab(t);
  // 矢印キー・Home・End でタブを移る
  $('tabs').addEventListener('keydown', (e) => {
    const i = TAB_IDS.indexOf(active);
    const keys: Record<string, number> = { ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: TAB_IDS.length - 1 };
    if (!(e.key in keys)) return;
    e.preventDefault();
    const id = TAB_IDS[(keys[e.key] + TAB_IDS.length) % TAB_IDS.length];
    selectTab(id);
    $(`tab-${id}`).focus();
  });
  for (const b of document.querySelectorAll<HTMLButtonElement>('[data-goto]')) {
    b.onclick = () => selectTab(b.dataset.goto as TabId);
  }
  window.addEventListener('hashchange', () => selectTab(tabFromHash()));
}

/** ドラッグ＆ドロップはページ全体で受け付け、表示中のタブに渡す */
function bindDrop(): void {
  let depth = 0;
  window.addEventListener('dragenter', (e) => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    depth++;
    document.body.classList.add('dragging');
  });
  window.addEventListener('dragover', (e) => {
    if (!e.dataTransfer?.types.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });
  window.addEventListener('dragleave', () => {
    if (--depth <= 0) {
      depth = 0;
      document.body.classList.remove('dragging');
    }
  });
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    document.body.classList.remove('dragging');
    tabs[active].addFiles([...(e.dataTransfer?.files ?? [])]);
  });
}

async function init(): Promise<void> {
  bindTabs();
  bindDrop();
  $('open-help').onclick = () => $<HTMLDialogElement>('help').showModal();
  selectTab(tabFromHash());
  const isolated = await ensureCrossOriginIsolation();
  if (!isolated) {
    const banner = $('banner');
    banner.hidden = false;
    banner.textContent =
      'このブラウザ環境では RAW の読み込み機能が使えません（JPEG / PNG は使えます）。通常ウィンドウの最新の Chrome / Edge / Firefox / Safari でお試しください。';
  }
}

void init();
