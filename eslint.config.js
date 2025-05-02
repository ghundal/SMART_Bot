// eslint.config.js - Flat config format
module.exports = [
  {
    // Expanded ignores section to cover your specific case
    ignores: [
      // Next.js build directories - with multiple variations to catch all cases
      '**/.next/**',
      '**/src/frontend/.next/**',
      '**/frontend/.next/**',
      '**/E115_SMART/src/frontend/.next/**',
      '/next/home/gurpreet/Desktop/Spring2025/CSCI115/Project/E115_SMART/src/frontend/.next/**',
      // Other common directories to ignore
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**'
    ],
  },
  {
    files: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx'],
    languageOptions: {
      parser: require('@typescript-eslint/parser'),
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
      },
      globals: {
        browser: true,
        node: true,
        es6: true,
      },
    },
    plugins: {
      '@typescript-eslint': require('@typescript-eslint/eslint-plugin'),
    },
    rules: {
      // TypeScript specific rules
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
      '@typescript-eslint/no-unused-vars': ['error', {
        'argsIgnorePattern': '^_',
        'varsIgnorePattern': '^_'
      }],

      // General best practices
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'prefer-const': 'error',
      'no-var': 'error',
      'eqeqeq': ['error', 'always'],
      'curly': ['error', 'all'],
      'no-duplicate-imports': 'error',
      'no-multiple-empty-lines': ['error', { 'max': 1, 'maxEOF': 1 }],
      'sort-imports': ['error', {
        'ignoreCase': true,
        'ignoreDeclarationSort': true,
        'ignoreMemberSort': false
      }]
    },
  },
  // Prettier rules instead of using extends
  {
    files: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx'],
    rules: {
      'arrow-body-style': 'off',
      'prefer-arrow-callback': 'off',
    },
  },
];
