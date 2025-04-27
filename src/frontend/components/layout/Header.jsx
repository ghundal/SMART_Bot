'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAuth } from '../../context/AuthContext';
import styles from './Header.module.css';

const navItems = [
  { name: 'About', path: '/about' },
  { name: 'Chat', path: '/chat' },
  { name: 'Reports', path: '/reports' },
];

export default function Header() {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const { user, logout } = useAuth();

  // If no user, don't render the header
  if (!user) return null;

  return (
    <>
      <header className={styles.header}>
        <nav className={styles.nav}>
          <Link href="/about" className={styles.logo}>
            <span>SMART</span>
          </Link>

          <div className={styles.navLinks}>
            {navItems.map((item) => (
              <Link
                key={item.name}
                href={item.path}
                className={`${styles.navLink} ${pathname === item.path ? styles.active : ''}`}
              >
                <span className={styles.linkText}>{item.name}</span>
              </Link>
            ))}

            <button onClick={logout} className={styles.logoutButton}>
              Logout
            </button>
          </div>

          <button
            className={`${styles.menuButton} ${isMobileMenuOpen ? styles.active : ''}`}
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          >
            <span></span>
            <span></span>
            <span></span>
          </button>
        </nav>
      </header>

      <div className={`${styles.mobileMenu} ${isMobileMenuOpen ? styles.active : ''}`}>
        {navItems.map((item) => (
          <Link
            key={item.name}
            href={item.path}
            className={`${styles.navLink} ${pathname === item.path ? styles.active : ''}`}
            onClick={() => setIsMobileMenuOpen(false)}
          >
            <span className={styles.linkText}>{item.name}</span>
          </Link>
        ))}
        <button
          onClick={() => {
            logout();
            setIsMobileMenuOpen(false);
          }}
          className={`${styles.navLink} ${styles.logoutLink}`}
        >
          Logout
        </button>
      </div>
    </>
  );
}
