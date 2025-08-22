import React from 'react';
import { motion } from 'framer-motion';

const StatCard = ({ title, value, icon: Icon, color, change, changeType }) => {
  const getColorClasses = (color) => {
    const colors = {
      primary: {
        bg: 'bg-primary-50',
        text: 'text-primary-600',
        border: 'border-primary-200',
        iconBg: 'bg-primary-100'
      },
      success: {
        bg: 'bg-success-50',
        text: 'text-success-600',
        border: 'border-success-200',
        iconBg: 'bg-success-100'
      },
      warning: {
        bg: 'bg-warning-50',
        text: 'text-warning-600',
        border: 'border-warning-200',
        iconBg: 'bg-warning-100'
      },
      error: {
        bg: 'bg-error-50',
        text: 'text-error-600',
        border: 'border-error-200',
        iconBg: 'bg-error-100'
      }
    };
    return colors[color] || colors.primary;
  };

  const getChangeColor = (changeType) => {
    switch (changeType) {
      case 'positive':
        return 'text-success-600';
      case 'negative':
        return 'text-error-600';
      case 'neutral':
        return 'text-gray-600';
      default:
        return 'text-gray-600';
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
      className={`bg-white rounded-xl shadow-sm border ${colorClasses.border} p-6 hover:shadow-md transition-all duration-200`}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900 mb-2">{value}</p>
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
