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
      <div className="flex items-center justify-center min-h-screen pt-20">
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
      <div className="flex items-center justify-center min-h-screen pt-20">
        <div className="text-center">
          <div className="text-error-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">Error Loading Dashboard</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
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
      value: dashboardStats?.best_model || 'EfficientNet-B0',
      icon: Award,
      color: 'primary',
      change: 'Top performer',
      changeType: 'neutral'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 pt-6">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Dashboard Overview</h1>
        <p className="text-gray-600 dark:text-gray-400">Monitor your AI models and system performance</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
          >
            <StatCard {...stat} />
          </motion.div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Model Performance Chart */}
        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.4 }}
          >
            <ModelPerformanceChart data={dashboardStats?.model_performance} />
          </motion.div>
        </div>

        {/* Quick Actions */}
        <div>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.5 }}
          >
            <QuickActions />
          </motion.div>
        </div>
      </div>

      {/* Recent Predictions */}
      <div className="mt-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.6 }}
        >
          <RecentPredictions predictions={dashboardStats?.recent_predictions} />
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;
