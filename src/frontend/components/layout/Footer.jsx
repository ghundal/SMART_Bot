'use client'
import { usePathname } from 'next/navigation';
import styles from './Footer.module.css';

export default function Footer() {
  return (
    <footer className="bg-white">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 border-t border-gray-200">
        <p className="text-center text-sm text-gray-500">
          &copy; {new Date().getFullYear()} SMART. All rights reserved.
        </p>
      </div>
    </footer>
  );
}