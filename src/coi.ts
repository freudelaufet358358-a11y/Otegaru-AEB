// クロスオリジン分離（SharedArrayBuffer）を有効にする。
// 開発サーバーや対応ホスティングではヘッダで有効になっているので何もしない。
// ヘッダを付けられない環境ではサービスワーカーを登録して 1 回だけ再読み込みする。

const KEY = 'otegaru-aeb:coi-reload';

export async function ensureCrossOriginIsolation(): Promise<boolean> {
  if (crossOriginIsolated) {
    sessionStorage.removeItem(KEY);
    return true;
  }
  if (!('serviceWorker' in navigator) || !isSecureContext) return false;
  if (sessionStorage.getItem(KEY)) {
    // 再読み込みしても有効にならなかった（プライベートモードなど）
    sessionStorage.removeItem(KEY);
    return false;
  }
  try {
    await navigator.serviceWorker.register(new URL('coi-serviceworker.js', document.baseURI).href);
    await navigator.serviceWorker.ready;
    sessionStorage.setItem(KEY, '1');
    location.reload();
    await new Promise(() => {});
  } catch (e) {
    console.warn('サービスワーカーを登録できませんでした', e);
  }
  return false;
}
