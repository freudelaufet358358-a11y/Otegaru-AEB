// LibRaw (WASM) をビルドを通さずそのまま静的ファイルとして配信するためにコピーする。
// libraw-wasm は内部で import.meta.url 基準に worker.js / libraw.js / libraw.wasm を読み込むため、
// バンドラに通すとパスが壊れやすい。public/vendor/libraw/ に置いて実行時に動的 import する。
import { copyFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'node_modules', 'libraw-wasm', 'dist');
const dest = join(root, 'public', 'vendor', 'libraw');
const files = ['index.js', 'worker.js', 'libraw.js', 'libraw.wasm'];

if (!existsSync(src)) {
  console.error('libraw-wasm が見つかりません。先に npm install を実行してください。');
  process.exit(1);
}
mkdirSync(dest, { recursive: true });
for (const f of files) copyFileSync(join(src, f), join(dest, f));
console.log(`LibRaw WASM を ${dest} にコピーしました`);
