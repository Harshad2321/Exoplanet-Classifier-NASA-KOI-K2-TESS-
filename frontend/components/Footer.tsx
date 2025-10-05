import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer id="contact-us" className="w-full max-w-5xl mx-auto py-8 px-4 text-slate-500 text-sm border-t border-slate-800">
      <h3 className="text-xl font-semibold text-slate-300 mb-6 text-center">Contact Us</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-3xl mx-auto">
        {/* Parth's Info */}
        <div className="text-center">
          <h4 className="text-base font-semibold text-slate-400 mb-3">Parth Koshti</h4>
          <div className="space-y-2">
            <p className="flex items-center justify-center">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
                <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
              </svg>
              <a href="mailto:Parthkoshti2606@gmail.com" className="hover:text-indigo-400 transition-colors">
                Parthkoshti2606@gmail.com
              </a>
            </p>
            <p className="flex items-center justify-center">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z" clipRule="evenodd" />
              </svg>
              <a href="https://github.com/Parth2619" target="_blank" rel="noopener noreferrer" className="hover:text-indigo-400 transition-colors">
                github.com/Parth2619
              </a>
            </p>
          </div>
        </div>

        {/* Harshad's Info */}
        <div className="text-center">
          <h4 className="text-base font-semibold text-slate-400 mb-3">Harshad Agrawal</h4>
          <div className="space-y-2">
            <p className="flex items-center justify-center">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
                <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
              </svg>
              <a href="mailto:Harshad.agrawal2005@gmail.com" className="hover:text-indigo-400 transition-colors">
                Harshad.agrawal2005@gmail.com
              </a>
            </p>
            <p className="flex items-center justify-center">
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z" clipRule="evenodd" />
              </svg>
              <a href="https://github.com/Harshad2321" target="_blank" rel="noopener noreferrer" className="hover:text-indigo-400 transition-colors">
                github.com/Harshad2321
              </a>
            </p>
          </div>
        </div>
      </div>

      {/* Copyright */}
      <div className="mt-8 pt-6 border-t border-slate-800 text-center text-xs text-slate-600">
        <p>© 2025 NASA Exoplanet Classifier. Built with passion for space exploration. 🚀🌟</p>
      </div>
    </footer>
  );
};