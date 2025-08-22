import React from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Image, 
  BarChart3, 
  Settings, 
  Play, 
  Download,
  Upload,
  RefreshCw
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const QuickActions = () => {
  const navigate = useNavigate();

  const actions = [
    {
      title: 'Test Image',
      description: 'Test models with random images',
      icon: Image,
      color: 'primary',
      action: () => navigate('/test-image'),
      badge: 'New'
    },
    {
      title: 'Model Training',
      description: 'Train new models or fine-tune existing ones',
      icon: Brain,
      color: 'success',
      action: () => navigate('/training'),
      badge: 'Active'
    },
    {
      title: 'Performance Analytics',
      description: 'Detailed model performance metrics',
      icon: BarChart3,
      color: 'warning',
      action: () => navigate('/analytics'),
      disabled: false
    },
    {
      title: 'Model Management',
      description: 'Manage and configure models',
      icon: Settings,
      color: 'secondary',
      action: () => navigate('/models'),
      disabled: false
    },
    {
      title: 'Start Inference',
      description: 'Begin real-time image classification',
      icon: Play,
      color: 'success',
      action: () => navigate('/inference'),
      disabled: false
    },
    {
      title: 'Export Results',
      description: 'Download prediction results and reports',
      icon: Download,
      color: 'primary',
      action: () => navigate('/export'),
      disabled: false
    }
  ];

  const getColorClasses = (color) => {
    const colors = {
      primary: {
        bg: 'bg-primary-50 dark:bg-primary-900/20',
        text: 'text-primary-600 dark:text-primary-400',
        border: 'border-primary-200 dark:border-primary-700',
        iconBg: 'bg-primary-100 dark:bg-primary-800',
        hover: 'hover:bg-primary-100 dark:hover:bg-primary-900/30'
      },
      success: {
        bg: 'bg-success-50 dark:bg-success-900/20',
        text: 'text-success-600 dark:text-success-400',
        border: 'border-success-200 dark:border-success-700',
        iconBg: 'bg-success-100 dark:bg-success-800',
        hover: 'hover:bg-success-100 dark:hover:bg-success-900/30'
      },
      warning: {
        bg: 'bg-warning-50 dark:bg-warning-900/20',
        text: 'text-warning-600 dark:text-warning-400',
        border: 'border-warning-200 dark:border-warning-700',
        iconBg: 'bg-warning-100 dark:bg-warning-800',
        hover: 'hover:bg-warning-100 dark:hover:bg-warning-900/30'
      },
      secondary: {
        bg: 'bg-secondary-50 dark:bg-secondary-900/20',
        text: 'text-secondary-600 dark:text-secondary-400',
        border: 'border-secondary-200 dark:border-secondary-700',
        iconBg: 'bg-secondary-100 dark:bg-secondary-800',
        hover: 'hover:bg-secondary-100 dark:hover:bg-secondary-900/30'
      }
    };
    return colors[color] || colors.primary;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
        ⚡ Quick Actions
      </h3>
      
      <div className="grid grid-cols-1 gap-3">
        {actions.map((action, index) => {
          const Icon = action.icon;
          const colorClasses = getColorClasses(action.color);
          
          return (
            <motion.button
              key={action.title}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={action.action}
              disabled={action.disabled}
              className={`w-full p-3 rounded-lg border transition-all duration-200 ${colorClasses.bg} ${colorClasses.border} ${colorClasses.hover} ${action.disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-lg ${colorClasses.iconBg}`}>
                  <Icon className={`w-5 h-5 ${colorClasses.text}`} />
                </div>
                <div className="flex-1 text-left">
                  <div className="flex items-center">
                    <span className={`font-medium ${colorClasses.text}`}>{action.title}</span>
                    {action.badge && (
                      <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-primary-100 dark:bg-primary-800 text-primary-800 dark:text-primary-200">
                        {action.badge}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{action.description}</p>
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
};

export default QuickActions;
