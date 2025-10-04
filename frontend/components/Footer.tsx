import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer id="contact-us" className="w-full max-w-4xl mx-auto py-6 px-4 text-center text-slate-500 text-sm">
        <h3 className="text-lg font-semibold text-slate-300 mb-2">Contact Us</h3>
        <p>For inquiries, please reach out at <a href="mailto:contact@exoplanetclassifier.com" className="hover:text-slate-300 transition-colors underline">contact@exoplanetclassifier.com</a>.</p>
    </footer>
  );
};