import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ModelPerformanceChart = ({ data }) => {
  // Transform data for the chart
  const chartData = data.map(item => ({
    name: item.model_name,
    accuracy: parseFloat(item.accuracy),
    f1Score: parseFloat(item.macro_f1),
    precision: parseFloat(item.precision),
    recall: parseFloat(item.recall)
  }));

  const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900">{label}</p>
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

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <div className="text-center">
          <div className="text-4xl mb-2">📊</div>
          <p>No performance data available</p>
        </div>
      </div>
    );
  }

  return (
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
      
      {/* Legend */}
      <div className="flex items-center justify-center mt-4 space-x-6">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-primary-500 rounded mr-2"></div>
          <span className="text-sm text-gray-600">Accuracy</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-success-500 rounded mr-2"></div>
          <span className="text-sm text-gray-600">F1-Score</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-warning-500 rounded mr-2"></div>
          <span className="text-sm text-gray-600">Precision</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-error-500 rounded mr-2"></div>
          <span className="text-sm text-gray-600">Recall</span>
        </div>
      </div>
    </div>
  );
};

export default ModelPerformanceChart;
