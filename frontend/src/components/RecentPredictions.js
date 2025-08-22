import React from 'react';
import { motion } from 'framer-motion';
import { Clock, CheckCircle, XCircle, Brain } from 'lucide-react';

const RecentPredictions = ({ predictions }) => {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-500">
        <div className="text-center">
          <Brain className="w-8 h-8 mx-auto mb-2 text-gray-300" />
          <p>No recent predictions</p>
        </div>
      </div>
    );
  }

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return date.toLocaleDateString();
  };

  const getStatusIcon = (predictedClass, actualClass) => {
    if (predictedClass === actualClass) {
      return <CheckCircle className="w-4 h-4 text-success-500" />;
    }
    return <XCircle className="w-4 h-4 text-error-500" />;
  };

  const getStatusColor = (predictedClass, actualClass) => {
    if (predictedClass === actualClass) {
      return 'text-success-600';
    }
    return 'text-error-600';
  };

  return (
    <div className="space-y-3">
      {predictions.slice(0, 5).map((prediction, index) => (
        <motion.div
          key={prediction.prediction_id || index}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
          className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-8 h-8 bg-primary-100 rounded-full">
              <Brain className="w-4 h-4 text-primary-600" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-medium text-gray-900">
                  {prediction.predicted_class}
                </span>
                {getStatusIcon(prediction.predicted_class, prediction.actual_class)}
              </div>
              <div className="flex items-center space-x-4 text-sm text-gray-500">
                <span>{prediction.model_name}</span>
                <span>•</span>
                <span className={getStatusColor(prediction.predicted_class, prediction.actual_class)}>
                  {(prediction.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-3 text-sm text-gray-500">
            <div className="flex items-center">
              <Clock className="w-4 h-4 mr-1" />
              <span>{formatTime(prediction.timestamp)}</span>
            </div>
            <div className="text-xs bg-gray-200 px-2 py-1 rounded">
              {prediction.processing_time.toFixed(2)}s
            </div>
          </div>
        </motion.div>
      ))}
      
      {predictions.length > 5 && (
        <div className="text-center pt-2">
          <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
            View all {predictions.length} predictions
          </button>
        </div>
      )}
    </div>
  );
};

export default RecentPredictions;
