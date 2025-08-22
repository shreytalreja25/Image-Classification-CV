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
        bg: 'bg-primary-50',
        text: 'text-primary-600',
        border: 'border-primary-200',
        iconBg: 'bg-primary-100',
        hover: 'hover:bg-primary-100'
      },
      success: {
        bg: 'bg-success-50',
        text: 'text-success-600',
        border: 'border-success-200',
        iconBg: 'bg-success-100',
        hover: 'hover:bg-success-100'
      },
      warning: {
        bg: 'bg-warning-50',
        text: 'text-warning-600',
        border: 'border-warning-200',
        iconBg: 'bg-warning-100',
        hover: 'hover:bg-warning-100'
      },
      secondary: {
        bg: 'bg-secondary-50',
        text: 'text-secondary-600',
        border: 'border-secondary-200',
        iconBg: 'bg-secondary-100',
        hover: 'hover:bg-secondary-100'
      }
    };
    return colors[color] || colors.primary;
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <RefreshCw className="w-5 h-5 mr-2 text-primary-500" />
        Quick Actions
      </h3>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {actions.map((action, index) => {
          const colorClasses = getColorClasses(action.color);
          
          return (
            <motion.button
              key={action.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              whileHover={{ y: -2, scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={action.action}
              disabled={action.disabled}
              className={`p-4 rounded-lg border ${colorClasses.border} ${colorClasses.bg} ${colorClasses.hover} transition-all duration-200 text-left disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center mb-2">
                    <div className={`p-2 rounded-lg ${colorClasses.iconBg} mr-3`}>
                      <action.icon className={`w-5 h-5 ${colorClasses.text}`} />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900">{action.title}</h4>
                      {action.badge && (
                        <span className="inline-block px-2 py-1 text-xs font-medium bg-primary-100 text-primary-800 rounded-full ml-2">
                          {action.badge}
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="text-sm text-gray-600">{action.description}</p>
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* Additional Info */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center text-sm text-gray-600">
          <Upload className="w-4 h-4 mr-2" />
          <span>Drag and drop images here for quick classification</span>
        </div>
      </div>
    </div>
  );
};

export default QuickActions;
