// This file configures ESLint, the code linter.
// It sets up recommended rules for JavaScript (js.configs.recommended),
// React Hooks (reactHooks.configs['recommended-latest']),
// and Vite-specific React Refresh (reactRefresh.configs.vite).
// It defines the environment (browser globals), JavaScript version (latest),
// enables JSX parsing, and sets the module type.
// The 'no-unused-vars' rule is configured to ignore variables starting with
// an underscore or uppercase letter (common for unused function arguments or constants).
// This is a standard and well-configured ESLint setup for this project type.
// No improvements needed unless you have specific linting preferences.

import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']), // Ignore the build output directory
  {
    files: ['**/*.{js,jsx}'], // Apply these rules to JS and JSX files
    extends: [
      js.configs.recommended, // ESLint recommended rules
      reactHooks.configs['recommended-latest'], // Latest React Hooks rules
      reactRefresh.configs.vite, // Vite HMR rules
    ],
    languageOptions: {
      ecmaVersion: 2020, // Use modern JavaScript features
      globals: globals.browser, // Define browser global variables (like 'window', 'document')
      parserOptions: {
        ecmaVersion: 'latest', // Use the latest ECMAScript version
        ecmaFeatures: { jsx: true }, // Enable JSX parsing
        sourceType: 'module', // Use ES modules (import/export)
      },
    },
    rules: {
      // Allow unused variables if they start with _ or are ALL_CAPS
      'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }],
      // Add any project-specific rules here if needed
    },
  },
])