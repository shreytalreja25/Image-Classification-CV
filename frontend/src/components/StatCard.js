import React from 'react';
import { motion } from 'framer-motion';

const StatCard = ({ title, value, icon: Icon, color, change, changeType }) => {
  const getColorClasses = (color) => {
    const colors = {
      primary: {
        bg: 'bg-primary-50 dark:bg-primary-900/20',
        text: 'text-primary-600 dark:text-primary-400',
        border: 'border-primary-200 dark:border-primary-700',
        iconBg: 'bg-primary-100 dark:bg-primary-800'
      },
      success: {
        bg: 'bg-success-50 dark:bg-success-900/20',
        text: 'text-success-600 dark:text-success-400',
        border: 'border-success-200 dark:border-success-700',
        iconBg: 'bg-success-100 dark:bg-success-800'
      },
      warning: {
        bg: 'bg-warning-50 dark:bg-warning-900/20',
        text: 'text-warning-600 dark:text-warning-400',
        border: 'border-warning-200 dark:border-warning-700',
        iconBg: 'bg-warning-100 dark:bg-warning-800'
      },
      error: {
        bg: 'bg-error-50 dark:bg-error-900/20',
        text: 'text-error-600 dark:text-error-400',
        border: 'border-error-200 dark:border-error-700',
        iconBg: 'bg-error-100 dark:bg-error-800'
      }
    };
    return colors[color] || colors.primary;
  };

  const getChangeColor = (changeType) => {
    switch (changeType) {
      case 'positive':
        return 'text-success-600 dark:text-success-400';
      case 'negative':
        return 'text-error-600 dark:text-error-400';
      case 'neutral':
        return 'text-gray-600 dark:text-gray-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  const getChangeIcon = (changeType) => {
    switch (changeType) {
      case 'positive':
        return '↗';
      case 'negative':
        return '↘';
      case 'neutral':
        return '→';
      default:
        return '→';
    }
  };

  const colorClasses = getColorClasses(color);

  return (
    <motion.div
      whileHover={{ y: -5, scale: 1.02 }}
      transition={{ duration: 0.2 }}
      className={`bg-white dark:bg-gray-800 rounded-xl shadow-sm border ${colorClasses.border} p-6 hover:shadow-md transition-all duration-200`}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900 dark:text-white mb-2">{value}</p>
          {change && (
            <div className={`flex items-center text-sm ${getChangeColor(changeType)}`}>
              <span className="mr-1">{getChangeIcon(changeType)}</span>
              <span className="font-medium">{change}</span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-lg ${colorClasses.iconBg}`}>
          <Icon className={`w-6 h-6 ${colorClasses.text}`} />
        </div>
      </div>
    </motion.div>
  );
};

export default StatCard;
