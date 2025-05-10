/** @type {import('jest').Config} */
const config = {
  testEnvironment: 'jsdom',
  transform: {
    '^.+\\.jsx?$': 'babel-jest',
  },
  // Look for tests in any location with these patterns
  testMatch: [
    '**/__tests__/**/*.js?(x)',
    '**/?(*.)+(spec|test).js?(x)',
    '**/tests/frontend/**/*.js',
  ],
  // Setup files for test environment
  setupFiles: ['./jest-setup.js'],
  // Directories to search for modules
  modulePaths: ['<rootDir>'],
  // Mock environment variables
  testEnvironmentOptions: {
    url: 'http://localhost/',
  },
  // Verbose output for debugging
  verbose: true,
};

module.exports = config;
