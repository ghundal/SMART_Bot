/**
 * Reports Service for the SMART application
 * Handles API calls to the reports endpoints
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9000';

/**
 * Get system stats
 */
export async function getSystemStats() {
  const response = await fetch(`${API_URL}/api/reports/system_stats`, {
    method: 'GET',
    credentials: 'include', // Important for cookies
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Error fetching system stats: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Get user count
 */
export async function getUserCount() {
  const response = await fetch(`${API_URL}/api/reports/users`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Error fetching user count: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Get query count for the specified number of days
 */
export async function getQueryCount(days = 30) {
  const response = await fetch(`${API_URL}/api/reports/queries?days=${days}`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Error fetching query count: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Get top documents
 */
export async function getTopDocuments(limit = 10) {
  const response = await fetch(`${API_URL}/api/reports/top_documents?limit=${limit}`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Error fetching top documents: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Get query activity for the specified number of days
 */
export async function getQueryActivity(days = 30) {
  const response = await fetch(`${API_URL}/api/reports/query_activity?days=${days}`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Error fetching query activity: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Get top keywords used in queries
 */
export async function getTopKeywords(limit = 20, minLength = 4) {
  const response = await fetch(`${API_URL}/api/reports/top_keywords?limit=${limit}&min_length=${minLength}`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Error fetching top keywords: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Get top phrases used in queries
 */
export async function getTopPhrases(limit = 10) {
  const response = await fetch(`${API_URL}/api/reports/top_phrases?limit=${limit}`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Error fetching top phrases: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Get most active users
 */
export async function getUserActivity(limit = 10) {
  const response = await fetch(`${API_URL}/api/reports/user_activity?limit=${limit}`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Error fetching user activity: ${response.statusText}`);
  }

  return await response.json();
}

/**
 * Get daily active users for the specified number of days
 */
export async function getDailyActiveUsers(days = 30) {
  const response = await fetch(`${API_URL}/api/reports/daily_active_users?days=${days}`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Error fetching daily active users: ${response.statusText}`);
  }

  return await response.json();
}

const ReportsService = {
  getSystemStats,
  getUserCount,
  getQueryCount,
  getTopDocuments,
  getQueryActivity,
  getTopKeywords,
  getTopPhrases,
  getUserActivity,
  getDailyActiveUsers
};

export default ReportsService;