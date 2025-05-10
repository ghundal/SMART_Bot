'use client';

import { createContext, useContext, useEffect, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';

// Create the auth context
const AuthContext = createContext();

// Auth provider component
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  // Check for user on mount
  useEffect(() => {
    // Check if there's a user cookie
    const checkAuth = async () => {
      try {
        // In a real app, verify with backend
        const userEmail = getCookie('user_email');
        if (userEmail) {
          setUser({ email: userEmail });
        } else {
          setUser(null);
          // Redirect to login if not on login page
          const publicPaths = ['/', '/login'];
          if (!publicPaths.includes(pathname)) {
            router.push('/');
          }
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        setUser(null);
      } finally {
        setLoading(false);
      }
    };
    checkAuth();
  }, [pathname, router]);

  // Logout function
  const logout = () => {
    // Clear cookies (requires backend coordination)
    document.cookie = 'access_token=; Max-Age=0; path=/; samesite=lax; secure';
    document.cookie = 'user_email=; Max-Age=0; path=/; samesite=lax; secure';
    setUser(null);
    router.push('/');
  };

  // Helper function to get cookie value
  function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) {
      return parts.pop().split(';').shift();
    }
    return null;
  }

  return <AuthContext.Provider value={{ user, loading, logout }}>{children}</AuthContext.Provider>;
}

// Custom hook to use the auth context
export const useAuth = () => useContext(AuthContext);
