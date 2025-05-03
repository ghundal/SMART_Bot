/**
 * DataService.js
 * Handles all API calls to the backend
 */

// Base API URL from environment variables
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9000';

/**
 * Get session ID from localStorage or generate a new one
 */
const getSessionId = () => {
  if (typeof window === 'undefined') {
    return null;
  }
  let sessionId = localStorage.getItem('session_id');
  if (!sessionId) {
    sessionId =
      Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    localStorage.setItem('session_id', sessionId);
  }
  return sessionId;
};

/**
 * Generate a UUID for messages
 */
export const uuid = () => {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
};

/**
 * Common fetch function with authentication headers
 */
const fetchWithAuth = async (endpoint, options = {}) => {
  const sessionId = getSessionId();

  // Ensure we have valid headers
  const headers = {
    'Content-Type': 'application/json',
    'X-Session-ID': sessionId,
    ...(options.headers || {})
  };

  const defaultOptions = {
    credentials: 'include',
    headers
  };

  // Make the API call
  const response = await fetch(`${API_URL}${endpoint}`, {
    ...defaultOptions,
    ...options
  });

  // Handle errors
  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error');
    console.error(`API error: ${response.status} ${response.statusText} - ${errorText}`);
    throw new Error(`API call failed: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return response.json();
};

/**
 * Get recent chats
 */
export const GetChats = async (limit = null) => {
  let endpoint = '/api/chats';
  if (limit) {
    endpoint += `?limit=${limit}`;
  }
  const data = await fetchWithAuth(endpoint);
  return { data };
};

/**
 * Get a specific chat by ID
 */
export const GetChat = async (model, chatId) => {
  const data = await fetchWithAuth(`/api/chats/${chatId}`);
  return { data };
};

/**
 * Start a new chat with LLM
 */
export const StartChatWithLLM = async (model, message) => {
  console.log(`Starting new chat with model: ${model}, message: ${message.content}`);
  const data = await fetchWithAuth('/api/chats', {
    method: 'POST',
    body: JSON.stringify({
      content: message.content,
      model: model
    })
  });
  return { data };
};

/**
 * Continue an existing chat with LLM
 */
export const ContinueChatWithLLM = async (model, chatId, message) => {
  console.log(`Continuing chat: ${chatId} with model: ${model}, message: ${message.content}`);
  const data = await fetchWithAuth(`/api/chats/${chatId}`, {
    method: 'POST',
    body: JSON.stringify({
      content: message.content,
      model: model
    })
  });
  return { data };
};

/**
 * Delete a chat
 */
export const DeleteChat = async (chatId) => {
  const data = await fetchWithAuth(`/api/chats/${chatId}`, {
    method: 'DELETE'
  });
  return { data };
};

/**
 * Submit a query using the query endpoint
 */
export const SubmitQuery = async (model, chatId, question, sessionId) => {
  const data = await fetchWithAuth('/api/query', {
    method: 'POST',
    body: JSON.stringify({
      chat_id: chatId,
      question: question,
      model_name: model,
      session_id: sessionId || getSessionId()
    })
  });
  return { data };
};

// Alternative implementation of the same endpoints using the new format
const DataService = {
  GetChats,
  GetChat,
  StartChatWithLLM,
  ContinueChatWithLLM,
  DeleteChat,
  SubmitQuery,
  uuid,
  getSessionId
};

export default DataService;
