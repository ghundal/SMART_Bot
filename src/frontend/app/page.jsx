import Image from 'next/image';
import styles from './page.module.css';
import LoginButton from '@/components/auth/LoginButton';

export default function LoginPage() {
  return (
    <div className={styles.loginContainer}>
      <div className={styles.loginCard}>
        <div className={styles.logoContainer}>
          <Image
            src="/logo.png"
            alt="Harvard Logo"
            width={120}
            height={120}
            priority
            className={styles.logo}
          />
        </div>
        
        <h1 className={styles.title}>
          <span className={styles.highlight}>SMART</span>
        </h1>
        
        <p className={styles.subtitle}>
          Secure Management and Retrieval Technology
        </p>
        
        <p className={styles.description}>
          Access your organization's documents with intelligent AI-powered search.
        </p>
        
        <LoginButton className={styles.googleButton} />
      </div>
    </div>
  );
}