
import React from 'react';
import Image from 'next/image';

const AboutUs = () => {
  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-8">
        {/* Logo added here */}
        <div className="flex justify-center mb-6">
          <Image
            src="/logo.png"
            alt="SMART Logo"
            width={150}
            height={150}
            className="h-auto"
            priority
          />
        </div>
        <h1 className="text-4xl font-extrabold text-gray-900 sm:text-5xl sm:tracking-tight lg:text-6xl">
          About <span className="text-blue-600">SMART</span>
        </h1>
        <p className="mt-5 max-w-xl mx-auto text-xl text-gray-500">
          Empowering organizations with intelligent document retrieval
        </p>
      </div>
      <div className="prose prose-lg prose-blue mx-auto">
        <p className="mb-6">
          <strong>SMART</strong> (Secure Management and Retrieval Technology) is an intelligent document retrieval system built to help organizations
          quickly and securely access internal documents using advanced AI and Large Language Models (LLMs).
        </p>
        <p className="mb-6">
          Our platform combines enterprise-grade security, document-level attribution, and natural language interfaces to reduce time spent on
          internal queries and improve overall productivity.
        </p>
        <p className="mb-6">
          The SMART solution is designed by students at Harvard University as part of the CS/E-115 course, integrating real-world MLOps pipelines,
          vector search, RAG, and secure infrastructure using Google Cloud Platform.
        </p>
      </div>
      <div className="mt-12 text-center">
        <p className="text-sm font-medium text-gray-500">
          Built with ❤️ by Hellen Momoh, Tiffany Valdecantos, Gurpreet Hundal, and Spiro
        </p>
      </div>
    </div>
  );
};

export default AboutUs;