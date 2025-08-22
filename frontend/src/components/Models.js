import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Settings, Download, Upload, Play, Pause, Trash2 } from 'lucide-react';

const Models = () => {
  const models = [
    {
      id: 1,
      name: 'EfficientNet-B0',
      status: 'active',
      accuracy: 95.46,
      lastUpdated: '2025-01-15',
      size: '29.3 MB',
      type: 'Deep Learning'
    },
    {
      id: 2,
      name: 'MobileNetV2',
      status: 'active',
      accuracy: 93.00,
      lastUpdated: '2025-01-14',
      size: '14.0 MB',
      type: 'Deep Learning'
    },
    {
      id: 3,
      name: 'ResNet-18',
      status: 'active',
      accuracy: 90.87,
      lastUpdated: '2025-01-13',
      size: '44.7 MB',
      type: 'Deep Learning'
    },
    {
      id: 4,
      name: 'Random Forest',
      status: 'inactive',
      accuracy: 23.54,
      lastUpdated: '2025-01-10',
      size: '2.1 MB',
      type: 'Traditional ML'
    }
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'bg-success-100 text-success-800';
      case 'inactive':
        return 'bg-gray-100 text-gray-800';
      case 'training':
        return 'bg-warning-100 text-warning-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active':
        return '🟢';
      case 'inactive':
        return '⚪';
      case 'training':
        return '🟡';
      default:
        return '⚪';
    }
  };

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
            Model Management
          </h1>
          <p className="text-gray-600">
            Manage and configure your AI models for aerial landscape classification
          </p>
        </div>

        {/* Actions */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
          <div className="flex flex-wrap items-center gap-4">
            <button className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors flex items-center">
              <Upload className="w-4 h-4 mr-2" />
              Upload Model
            </button>
            <button className="px-4 py-2 bg-success-500 text-white rounded-lg hover:bg-success-600 transition-colors flex items-center">
              <Play className="w-4 h-4 mr-2" />
              Train New Model
            </button>
            <button className="px-4 py-2 bg-secondary-500 text-white rounded-lg hover:bg-secondary-600 transition-colors flex items-center">
              <Settings className="w-4 h-4 mr-2" />
              Configure
            </button>
          </div>
        </div>

        {/* Models Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models.map((model, index) => (
            <motion.div
              key={model.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-all duration-200"
            >
              {/* Model Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 mb-1">{model.name}</h3>
                  <p className="text-sm text-gray-500">{model.type}</p>
                </div>
                <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(model.status)}`}>
                  {getStatusIcon(model.status)} {model.status}
                </div>
              </div>

              {/* Model Stats */}
              <div className="space-y-3 mb-6">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Accuracy</span>
                  <span className="text-sm font-medium text-gray-900">{model.accuracy}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Size</span>
                  <span className="text-sm font-medium text-gray-900">{model.size}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Last Updated</span>
                  <span className="text-sm font-medium text-gray-900">{model.lastUpdated}</span>
                </div>
              </div>

              {/* Model Actions */}
              <div className="flex space-x-2">
                <button className="flex-1 px-3 py-2 bg-primary-50 text-primary-700 rounded-md hover:bg-primary-100 transition-colors text-sm font-medium">
                  <Play className="w-4 h-4 mr-1 inline" />
                  Activate
                </button>
                <button className="flex-1 px-3 py-2 bg-secondary-50 text-secondary-700 rounded-md hover:bg-secondary-100 transition-colors text-sm font-medium">
                  <Settings className="w-4 h-4 mr-1 inline" />
                  Configure
                </button>
                <button className="px-3 py-2 bg-gray-50 text-gray-700 rounded-md hover:bg-gray-100 transition-colors">
                  <Download className="w-4 h-4" />
                </button>
                <button className="px-3 py-2 bg-error-50 text-error-700 rounded-md hover:bg-error-100 transition-colors">
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Model Performance Summary */}
        <div className="mt-8 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Performance Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary-600">4</div>
              <div className="text-sm text-gray-600">Total Models</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-success-600">3</div>
              <div className="text-sm text-gray-600">Active Models</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-warning-600">95.46%</div>
              <div className="text-sm text-gray-600">Best Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-secondary-600">90.2 MB</div>
              <div className="text-sm text-gray-600">Total Size</div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Models;
