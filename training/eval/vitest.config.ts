import { defineConfig } from 'vitest/config';

// 評価用の設定（npm test には含めない）
export default defineConfig({
  assetsInclude: ['**/*.bin'],
  test: { include: ['training/eval/render_eval.ts'], testTimeout: 3_600_000 },
});
