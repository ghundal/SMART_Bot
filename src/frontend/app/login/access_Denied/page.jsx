'use client';
import Image from 'next/image';
import Link from 'next/link';
import styles from '../../../components/auth/access.module.css';

export default function AccessDenied() {
  return (
    <div className={styles.loginContainer}>
      <div className={styles.loginCard}>
        <div className={styles.logoContainer}>
          <Image
            src="/logo.png"
            alt="SMART Logo"
            width={120}
            height={120}
            priority
            className={styles.logo}
          />
        </div>
        <h1 className={styles.title}>
          <span className={styles.highlight}>Access Denied</span>
        </h1>
        <p className={styles.subtitle}>Secure Management and Retrieval Technology</p>

        <div className={styles.accessMessage}>
          <svg
            viewBox="0 0 24 24"
            width="48"
            height="48"
            className={styles.warningIcon}
          >
            <path
              d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 11c-.55 0-1-.45-1-1V8c0-.55.45-1 1-1s1 .45 1 1v4c0 .55-.45 1-1 1zm1 4h-2v-2h2v2z"
              fill="#EA4335"
            />
          </svg>

          <p className={styles.description}>
            You don't currently have access to the SMART system.
            <br /><br />
            Please contact our support team to request access:
          </p>

          <a href="mailto:smart@gmail.com" className={styles.contactEmail}>
            <svg
              viewBox="0 0 24 24"
              width="20"
              height="20"
              stroke="currentColor"
              strokeWidth="2"
              fill="none"
              className={styles.emailIcon}
            >
              <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
              <polyline points="22,6 12,13 2,6"></polyline>
            </svg>
            smart@gmail.com
          </a>
        </div>

        <Link href="/login" className={styles.returnButton}>
          Return to Login
        </Link>
      </div>
    </div>
  );
}
