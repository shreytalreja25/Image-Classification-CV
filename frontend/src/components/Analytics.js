import React from 'react';
import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Target, Clock, Brain, Activity } from 'lucide-react';

const Analytics = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-7xl mx-auto"
      >
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center">
            <BarChart3 className="w-8 h-8 mr-3 text-primary-500" />
            Performance Analytics
          </h1>
          <p className="text-gray-600">
            Detailed insights into model performance and classification metrics
          </p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center">
              <div className="p-2 bg-primary-100 rounded-lg">
                <Brain className="w-6 h-6 text-primary-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Models</p>
                <p className="text-2xl font-bold text-gray-900">4</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center">
              <div className="p-2 bg-success-100 rounded-lg">
                <Target className="w-6 h-6 text-success-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Best Accuracy</p>
                <p className="text-2xl font-bold text-gray-900">95.46%</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center">
              <div className="p-2 bg-warning-100 rounded-lg">
                <Activity className="w-6 h-6 text-warning-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Avg Processing</p>
                <p className="text-2xl font-bold text-gray-900">2.3s</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <div className="flex items-center">
              <div className="p-2 bg-secondary-100 rounded-lg">
                <Clock className="w-6 h-6 text-secondary-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Uptime</p>
                <p className="text-2xl font-bold text-gray-900">99.9%</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Performance Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Comparison</h3>
            <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
              <div className="text-center">
                <div className="text-4xl mb-2">📊</div>
                <p className="text-gray-600">Chart coming soon</p>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Accuracy Trends</h3>
            <div className="h-64 flex items-center justify-center bg-gray-50 rounded-lg">
              <div className="text-center">
                <div className="text-4xl mb-2">📈</div>
                <p className="text-gray-600">Trend chart coming soon</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Detailed Metrics */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Performance Metrics</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Precision</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Recall</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">F1-Score</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Processing Time</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">EfficientNet-B0</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">95.46%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">95.53%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">95.33%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">95.45%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">2.1s</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">MobileNetV2</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">93.00%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">93.10%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">93.00%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">93.02%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">1.8s</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">ResNet-18</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">90.87%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">91.00%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">90.87%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">90.83%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">2.5s</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Random Forest</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">23.54%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">22.07%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">23.54%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">22.07%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">0.3s</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Analytics;
