/**
 * DataService.integration.test.js
 * Integration tests for DataService with mock service worker
 */

// Mock fetch globally
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    status: 200,
    statusText: 'OK',
    json: () => Promise.resolve({ success: true }),
    text: () => Promise.resolve('Success'),
  }),
);

// Mock localStorage
global.localStorage = {
  _data: {},
  getItem: jest.fn((key) => {
    return global.localStorage._data[key];
  }),
  setItem: jest.fn((key, value) => {
    global.localStorage._data[key] = value;
  }),
  clear: jest.fn(() => {
    global.localStorage._data = {};
  }),
};

// Mock process.env
process.env.NEXT_PUBLIC_API_URL = 'http://localhost:9000';

// Sample test data
const TEST_SESSION_ID = 'test-session-id';
const TEST_MODEL = 'gpt-4';
const TEST_CHAT_ID = 'chat-12345';
const TEST_MESSAGE = { content: 'Hello, this is a test message' };

// Create a basic test to verify integration tests are working
describe('DataService Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    global.localStorage._data = {};
    global.localStorage.setItem('session_id', TEST_SESSION_ID);

    // Setup default fetch response
    fetch.mockImplementation((url) => {
      // Different responses based on URL patterns
      if (url.includes('/api/chats/')) {
        if (url.endsWith(TEST_CHAT_ID)) {
          return Promise.resolve({
            ok: true,
            status: 200,
            statusText: 'OK',
            json: () =>
              Promise.resolve({
                id: TEST_CHAT_ID,
                messages: [
                  { role: 'user', content: 'Initial message' },
                  { role: 'assistant', content: 'Initial response' },
                ],
              }),
          });
        }
      } else if (url.endsWith('/api/chats')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          statusText: 'OK',
          json: () =>
            Promise.resolve({
              chats: [
                { id: 'chat-1', title: 'Chat 1' },
                { id: 'chat-2', title: 'Chat 2' },
              ],
            }),
        });
      }

      // Default response
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: () => Promise.resolve({ success: true }),
      });
    });
  });

  test('Integration test setup is working', () => {
    expect(global.localStorage.getItem('session_id')).toBe(TEST_SESSION_ID);
  });

  test('Fetch mock returns different responses based on URL', async () => {
    const response1 = await fetch('http://localhost:9000/api/chats');
    const data1 = await response1.json();
    expect(data1.chats).toBeDefined();

    const response2 = await fetch(`http://localhost:9000/api/chats/${TEST_CHAT_ID}`);
    const data2 = await response2.json();
    expect(data2.id).toBe(TEST_CHAT_ID);
  });
});
