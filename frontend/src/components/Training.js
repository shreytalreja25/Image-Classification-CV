import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Play, Pause, Settings, BarChart3, Clock, Zap } from 'lucide-react';

const Training = () => {
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
            <Brain className="w-8 h-8 mr-3 text-primary-500" />
            Model Training
          </h1>
          <p className="text-gray-600">
            Train and fine-tune your AI models for aerial landscape classification
          </p>
        </div>

        {/* Training Status */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Training Status</h2>
            <div className="flex space-x-3">
              <button className="px-4 py-2 bg-success-500 text-white rounded-lg hover:bg-success-600 transition-colors flex items-center">
                <Play className="w-4 h-4 mr-2" />
                Start Training
              </button>
              <button className="px-4 py-2 bg-warning-500 text-white rounded-lg hover:bg-warning-600 transition-colors flex items-center">
                <Pause className="w-4 h-4 mr-2" />
                Pause
              </button>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-primary-50 rounded-lg">
              <div className="text-2xl font-bold text-primary-600">0</div>
              <div className="text-sm text-primary-700">Current Epoch</div>
            </div>
            <div className="text-center p-4 bg-success-50 rounded-lg">
              <div className="text-2xl font-bold text-success-600">0.00</div>
              <div className="text-sm text-success-700">Loss</div>
            </div>
            <div className="text-center p-4 bg-warning-50 rounded-lg">
              <div className="text-2xl font-bold text-warning-600">0.00%</div>
              <div className="text-sm text-warning-700">Accuracy</div>
            </div>
          </div>
        </div>

        {/* Training Configuration */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Settings className="w-5 h-5 mr-2 text-primary-500" />
              Training Configuration
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Model Architecture</label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500">
                  <option>EfficientNet-B0</option>
                  <option>MobileNetV2</option>
                  <option>ResNet-18</option>
                  <option>Custom Architecture</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Learning Rate</label>
                <input type="number" step="0.001" defaultValue="0.001" className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Batch Size</label>
                <input type="number" defaultValue="32" className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Epochs</label>
                <input type="number" defaultValue="100" className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-primary-500" />
              Training Progress
            </h3>
            <div className="space-y-4">
              <div className="text-center p-8 bg-gray-50 rounded-lg">
                <div className="text-4xl mb-2">📊</div>
                <p className="text-gray-600">No active training</p>
                <p className="text-sm text-gray-500">Start training to see progress</p>
              </div>
            </div>
          </div>
        </div>

        {/* Training History */}
        <div className="mt-8 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Training History</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Duration</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">EfficientNet-B0</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-success-100 text-success-800">Completed</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">95.46%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2h 15m</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2025-01-15</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">MobileNetV2</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-success-100 text-success-800">Completed</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">93.00%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">1h 45m</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2025-01-14</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Training;
