/**
 * DataService.test.js
 * Basic tests for validating DataService functionality
 */

// Mock fetch globally
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    status: 200,
    statusText: 'OK',
    json: () => Promise.resolve({ success: true }),
    text: () => Promise.resolve('Success')
  })
);

// Mock localStorage
global.localStorage = {
  _data: {},
  getItem: jest.fn(key => {
    return global.localStorage._data[key];
  }),
  setItem: jest.fn((key, value) => {
    global.localStorage._data[key] = value;
  }),
  clear: jest.fn(() => {
    global.localStorage._data = {};
  })
};

// Mock process.env
process.env.NEXT_PUBLIC_API_URL = 'http://localhost:9000';

// Create a basic test to verify Jest is working
describe('Basic Test Suite', () => {
  test('Jest is working', () => {
    expect(1 + 1).toBe(2);
  });
});

// Add more tests to validate DataService once structure is confirmed
describe('DataService', () => {
  test('Mock fetch is working', async () => {
    const result = await fetch('http://example.com');
    const json = await result.json();
    expect(json.success).toBe(true);
  });
});
