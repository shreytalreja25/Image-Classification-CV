import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  Menu, 
  X, 
  Home, 
  Brain, 
  Image, 
  BarChart3, 
  Settings, 
  Zap,
  ChevronRight,
  Sun,
  Moon
} from 'lucide-react';
import config from '../config/config';

const Navigation = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [wsStatus, setWsStatus] = useState('disconnected');
  const location = useLocation();
  const navigate = useNavigate();

  const navigationItems = [
    {
      name: 'Dashboard',
      path: '/',
      icon: Home,
      description: 'Overview and statistics'
    },
    {
      name: 'Test Image',
      path: '/test-image',
      icon: Image,
      description: 'Test models with images',
      badge: 'New'
    },
    {
      name: 'Models',
      path: '/models',
      icon: Brain,
      description: 'Model management'
    },
    {
      name: 'Training',
      path: '/training',
      icon: Brain,
      description: 'Model training'
    },
    {
      name: 'Analytics',
      path: '/analytics',
      icon: BarChart3,
      description: 'Performance metrics'
    },
    {
      name: 'Settings',
      path: '/settings',
      icon: Settings,
      description: 'Configuration'
    }
  ];

  useEffect(() => {
    // Check for saved dark mode preference
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    setDarkMode(savedDarkMode);
    
    // Apply dark mode class
    if (savedDarkMode) {
      document.documentElement.classList.add('dark');
    }
  }, []);

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('darkMode', newDarkMode);
    
    if (newDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  const getWsStatusColor = () => {
    switch (wsStatus) {
      case 'connected':
        return 'text-success-500';
      case 'connecting':
        return 'text-warning-500';
      case 'disconnected':
        return 'text-error-500';
      default:
        return 'text-gray-500';
    }
  };

  const getWsStatusIcon = () => {
    switch (wsStatus) {
      case 'connected':
        return '🟢';
      case 'connecting':
        return '🟡';
      case 'disconnected':
        return '🔴';
      default:
        return '⚪';
    }
  };

  return (
    <>
      {/* Mobile menu button */}
      <div className="lg:hidden fixed top-4 left-4 z-50">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-2 bg-white rounded-lg shadow-lg border border-gray-200 hover:bg-gray-50 transition-colors"
        >
          {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
        </button>
      </div>

      {/* Sidebar */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 lg:hidden"
          >
            <div
              className="fixed inset-0 bg-black bg-opacity-50"
              onClick={() => setIsOpen(false)}
            />
            <motion.div
              initial={{ x: -300 }}
              animate={{ x: 0 }}
              exit={{ x: -300 }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
              className="fixed left-0 top-0 h-full w-80 bg-white shadow-xl z-50"
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-8">
                  <h1 className="text-2xl font-bold text-gray-900">
                    {config.appName}
                  </h1>
                  <button
                    onClick={() => setIsOpen(false)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
                
                <nav className="space-y-2">
                  {navigationItems.map((item) => (
                    <NavigationItem
                      key={item.name}
                      item={item}
                      isActive={location.pathname === item.path}
                      onClick={() => {
                        navigate(item.path);
                        setIsOpen(false);
                      }}
                    />
                  ))}
                </nav>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Desktop sidebar */}
      <div className="hidden lg:flex lg:flex-col lg:w-80 lg:fixed lg:inset-y-0 lg:z-50">
        <div className="flex flex-col flex-grow bg-white border-r border-gray-200 pt-5 pb-4 overflow-y-auto">
          <div className="flex items-center flex-shrink-0 px-6 mb-8">
            <div className="flex items-center">
              <Brain className="w-8 h-8 text-primary-500 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">
                {config.appName}
              </h1>
            </div>
          </div>
          
          <nav className="flex-1 px-6 space-y-2">
            {navigationItems.map((item) => (
              <NavigationItem
                key={item.name}
                item={item}
                isActive={location.pathname === item.path}
                onClick={() => navigate(item.path)}
              />
            ))}
          </nav>

          {/* Bottom section */}
          <div className="px-6 py-4 border-t border-gray-200">
            {/* WebSocket Status */}
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm text-gray-600">Connection Status</span>
              <div className="flex items-center space-x-2">
                <span className="text-sm">{getWsStatusIcon()}</span>
                <span className={`text-sm ${getWsStatusColor()}`}>
                  {wsStatus}
                </span>
              </div>
            </div>

            {/* Dark Mode Toggle */}
            <button
              onClick={toggleDarkMode}
              className="flex items-center justify-between w-full p-3 text-sm text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <span>Dark Mode</span>
              {darkMode ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            </button>

            {/* App Version */}
            <div className="text-xs text-gray-500 text-center mt-4">
              v{config.appVersion}
            </div>
          </div>
        </div>
      </div>

      {/* Main content margin for desktop */}
      <div className="lg:ml-80" />
    </>
  );
};

const NavigationItem = ({ item, isActive, onClick }) => {
  const Icon = item.icon;
  
  return (
    <motion.button
      whileHover={{ x: 4 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`w-full flex items-center justify-between p-3 text-left rounded-lg transition-all duration-200 ${
        isActive
          ? 'bg-primary-50 text-primary-700 border border-primary-200'
          : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
      }`}
    >
      <div className="flex items-center">
        <Icon className={`w-5 h-5 mr-3 ${isActive ? 'text-primary-600' : 'text-gray-500'}`} />
        <div>
          <div className="flex items-center">
            <span className="font-medium">{item.name}</span>
            {item.badge && (
              <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                {item.badge}
              </span>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-1">{item.description}</p>
        </div>
      </div>
      <ChevronRight className={`w-4 h-4 ${isActive ? 'text-primary-600' : 'text-gray-400'}`} />
    </motion.button>
  );
};

export default Navigation;
