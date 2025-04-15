import Image from 'next/image';
import styles from '../../components/about/About.module.css';
import ProtectedRoute from '../../components/auth/ProtectedRoute';

export default function AboutUsPage() {
  return (
    <ProtectedRoute>
      <AboutContent />
    </ProtectedRoute>
  );
}

// Separate the content to keep it clean
function AboutContent() {
  return (
    <div className={styles.about}>
      <div className={styles.logoContainer}>
        <Image
          src="/logo.png"
          alt="Harvard Logo"
          width={150}
          height={150}
          className={styles.logo}
          priority
        />
      </div>
      
      <h1 className={styles.title}>
        About <span className={styles.highlight}>SMART</span>
      </h1>
      
      <div className={styles.underline}></div>
      
      <div className={styles.tagline}>
        Empowering organizations with intelligent document retrieval
      </div>
      
      <div className={styles.content}>
        <div className={styles.contentBox}>
          <p>
            <strong>SMART</strong> (Secure Management and Retrieval Technology) is an intelligent document retrieval system built to help organizations quickly and securely access internal documents using advanced AI and Large Language Models (LLMs).
          </p>
          <p>
            Our platform combines enterprise-grade security, document-level attribution, and natural language interfaces to reduce time spent on internal queries and improve overall productivity.
          </p>
          <p>
            The SMART solution is designed by students at Harvard University as part of the CS/E-115 course, integrating real-world MLOps pipelines, vector search, RAG, and secure infrastructure using Google Cloud Platform.
          </p>
        </div>
      </div>
      
      <div className={styles.footer}>
        <p>
          Built with <span className={styles.heart}>❤️</span> by Hellen Momoh, Tiffany Valdecantos, Gurpreet Hundal, and Spiro
        </p>
        <p className={styles.copyright}>
          © 2025 SMART. All rights reserved.
        </p>
      </div>
    </div>
  );
}