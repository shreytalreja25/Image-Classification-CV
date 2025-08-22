import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ModelPerformanceChart = ({ data }) => {
  // Default data when no data is provided
  const defaultData = [
    { model_name: 'EfficientNet-B0', accuracy: 94.2, macro_f1: 93.8, precision: 94.1, recall: 93.5 },
    { model_name: 'MobileNetV2', accuracy: 91.7, macro_f1: 91.3, precision: 91.6, recall: 91.0 },
    { model_name: 'ResNet-18', accuracy: 89.4, macro_f1: 89.0, precision: 89.3, recall: 88.7 }
  ];

  // Use provided data or default data
  const chartData = (data || defaultData).map(item => ({
    name: item.model_name,
    accuracy: parseFloat(item.accuracy || 0),
    f1Score: parseFloat(item.macro_f1 || 0),
    precision: parseFloat(item.precision || 0),
    recall: parseFloat(item.recall || 0)
  }));

  const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900 dark:text-white">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(2)}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center">
          📊 Model Performance
        </h2>
        <div className="flex space-x-2">
          <button className="px-3 py-1 text-sm bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 rounded-md hover:bg-primary-100 dark:hover:bg-primary-900/30 transition-colors">
            Accuracy
          </button>
          <button className="px-3 py-1 text-sm bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors">
            F1-Score
          </button>
        </div>
      </div>
      
      <div className="w-full h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
            <XAxis 
              dataKey="name" 
              stroke="#6B7280"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              stroke="#6B7280"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="accuracy" fill="#3B82F6" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Legend */}
      <div className="flex items-center justify-center mt-4 space-x-6">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-primary-500 rounded mr-2"></div>
          <span className="text-sm text-gray-600 dark:text-gray-400">Accuracy</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-success-500 rounded mr-2"></div>
          <span className="text-sm text-gray-600 dark:text-gray-400">F1-Score</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-warning-500 rounded mr-2"></div>
          <span className="text-sm text-gray-600 dark:text-gray-400">Precision</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-error-500 rounded mr-2"></div>
          <span className="text-sm text-gray-600 dark:text-gray-400">Recall</span>
        </div>
      </div>
    </div>
  );
};

export default ModelPerformanceChart;
