import { createRequire } from 'node:module';
import path from 'node:path';

const require = createRequire(import.meta.url);
// expo-module-scripts does not expose eslint.config.base.cjs via its "exports"
// map, so resolve it from the package's physical install location instead.
const pkgJsonPath = require.resolve('expo-module-scripts/package.json');
const base = require(path.join(path.dirname(pkgJsonPath), 'eslint.config.base.cjs'));

export default [
  ...base,
  {
    files: ['src/**/*.ts', 'src/**/*.tsx'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
    },
  },
];
