import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { wsService } from './services/api';
import config from './config/config';

// Components
import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import TestImage from './components/TestImage';
import Models from './components/Models';
import Training from './components/Training';
import Analytics from './components/Analytics';
import Settings from './components/Settings';

// Placeholder components for routes not yet implemented
const PlaceholderComponent = ({ title, description }) => (
  <div className="min-h-screen bg-gray-50 flex items-center justify-center">
    <div className="text-center">
      <div className="text-6xl mb-4">🚧</div>
      <h1 className="text-3xl font-bold text-gray-900 mb-2">{title}</h1>
      <p className="text-gray-600">{description}</p>
    </div>
  </div>
);

function App() {
  useEffect(() => {
    // Initialize WebSocket connection
    wsService.connect();
    
    // Set up event listeners
    wsService.on('connected', () => {
      console.log('WebSocket connected successfully');
    });
    
    wsService.on('trainingProgress', (data) => {
      console.log('Training progress:', data);
      // Handle training progress updates
    });
    
    wsService.on('predictionUpdate', (data) => {
      console.log('Prediction update:', data);
      // Handle prediction updates
    });
    
    wsService.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
    
    // Cleanup on unmount
    return () => {
      wsService.disconnect();
    };
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        
        <AnimatePresence mode="wait">
          <Routes>
            <Route 
              path="/" 
              element={
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Dashboard />
                </motion.div>
              } 
            />
            
            <Route 
              path="/test-image" 
              element={
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <TestImage />
                </motion.div>
              } 
            />
            
            <Route 
              path="/models" 
              element={
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Models />
                </motion.div>
              } 
            />
            
            <Route 
              path="/training" 
              element={
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Training />
                </motion.div>
              } 
            />
            
            <Route 
              path="/analytics" 
              element={
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Analytics />
                </motion.div>
              } 
            />
            
            <Route 
              path="/settings" 
              element={
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <Settings />
                </motion.div>
              } 
            />
            
            {/* Redirect unknown routes to dashboard */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AnimatePresence>
      </div>
    </Router>
  );
}

export default App;
