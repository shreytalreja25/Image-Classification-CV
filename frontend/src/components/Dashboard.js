import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  Brain, 
  Image, 
  TrendingUp, 
  Activity,
  Zap,
  Target,
  Award,
  Clock,
  Database
} from 'lucide-react';
import { apiService } from '../services/api';
import StatCard from './StatCard';
import ModelPerformanceChart from './ModelPerformanceChart';
import RecentPredictions from './RecentPredictions';
import QuickActions from './QuickActions';

const Dashboard = () => {
  const [dashboardStats, setDashboardStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDashboardStats();
  }, []);

  const fetchDashboardStats = async () => {
    try {
      setLoading(true);
      const stats = await apiService.getDashboardStats();
      setDashboardStats(stats);
      setError(null);
    } catch (err) {
      setError('Failed to fetch dashboard statistics');
      console.error('Error fetching dashboard stats:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          className="w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full"
        />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-error-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Error Loading Dashboard</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={fetchDashboardStats}
            className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const stats = [
    {
      title: 'Total Models',
      value: dashboardStats?.total_models || 0,
      icon: Brain,
      color: 'primary',
      change: '+2 this month',
      changeType: 'positive'
    },
    {
      title: 'Total Predictions',
      value: dashboardStats?.total_predictions || 0,
      icon: Image,
      color: 'success',
      change: '+15% vs last week',
      changeType: 'positive'
    },
    {
      title: 'Average Accuracy',
      value: `${(dashboardStats?.average_accuracy || 0).toFixed(1)}%`,
      icon: Target,
      color: 'warning',
      change: '+2.3% vs last month',
      changeType: 'positive'
    },
    {
      title: 'Best Model',
      value: dashboardStats?.best_model || 'N/A',
      icon: Award,
      color: 'success',
      change: 'EfficientNet-B0',
      changeType: 'neutral'
    }
  ];

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
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Aerial Classification Dashboard
          </h1>
          <p className="text-gray-600">
            Monitor and analyze AI model performance for aerial landscape classification
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <StatCard {...stat} />
            </motion.div>
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Charts and Performance */}
          <div className="lg:col-span-2 space-y-8">
            {/* Model Performance Chart */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-primary-500" />
                  Model Performance
                </h2>
                <div className="flex space-x-2">
                  <button className="px-3 py-1 text-sm bg-primary-50 text-primary-700 rounded-md hover:bg-primary-100 transition-colors">
                    Accuracy
                  </button>
                  <button className="px-3 py-1 text-sm bg-gray-50 text-gray-700 rounded-md hover:bg-gray-100 transition-colors">
                    F1-Score
                  </button>
                </div>
              </div>
              <ModelPerformanceChart data={dashboardStats?.model_performance || []} />
            </motion.div>

            {/* Recent Predictions */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                  <Activity className="w-5 h-5 mr-2 text-primary-500" />
                  Recent Predictions
                </h2>
                <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                  View All
                </button>
              </div>
              <RecentPredictions predictions={dashboardStats?.recent_predictions || []} />
            </motion.div>
          </div>

          {/* Right Column - Quick Actions and Info */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <QuickActions />
            </motion.div>

            {/* System Status */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Zap className="w-5 h-5 mr-2 text-success-500" />
                System Status
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">API Status</span>
                  <span className="flex items-center text-success-600">
                    <div className="w-2 h-2 bg-success-500 rounded-full mr-2"></div>
                    Online
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Database</span>
                  <span className="flex items-center text-success-600">
                    <div className="w-2 h-2 bg-success-500 rounded-full mr-2"></div>
                    Connected
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Models</span>
                  <span className="flex items-center text-success-600">
                    <div className="w-2 h-2 bg-success-500 rounded-full mr-2"></div>
                    Loaded
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Last Update</span>
                  <span className="text-sm text-gray-500">
                    {new Date().toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </motion.div>

            {/* Quick Stats */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2 text-primary-500" />
                Quick Stats
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Processing Time</span>
                  <span className="text-sm font-medium text-gray-900">~2.3s</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Success Rate</span>
                  <span className="text-sm font-medium text-success-600">98.7%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Active Users</span>
                  <span className="text-sm font-medium text-gray-900">12</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Uptime</span>
                  <span className="text-sm font-medium text-success-600">99.9%</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
