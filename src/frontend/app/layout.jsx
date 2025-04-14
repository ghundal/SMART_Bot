import './globals.css';
export const metadata = {
  title: 'SMART - Secure Management and Retrieval Technology',
  description: 'Intelligent document retrieval system built with advanced AI and LLMs',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16 items-center">
              <div className="flex items-center">
                <span className="text-xl font-bold text-gray-900">
                  <span className="text-blue-600">SMART</span>
                </span>
              </div>
              <nav className="flex space-x-8">
                <a href="/" className="text-gray-500 hover:text-gray-700">
                  Home
                </a>
                <a href="/about" className="text-gray-500 hover:text-gray-700">
                  About
                </a>
              </nav>
            </div>
          </div>
        </header>
        
        <main>
          {children}
        </main>
        
        <footer className="bg-white">
          <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 border-t border-gray-200">
            <p className="text-center text-sm text-gray-500">
              &copy; {new Date().getFullYear()} SMART. All rights reserved.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}