import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shuffle, 
  Brain, 
  Image as ImageIcon, 
  Target, 
  Clock, 
  BarChart3,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';
import { apiService } from '../services/api';
import config from '../config/config';

const TestImage = () => {
  const [selectedModel, setSelectedModel] = useState('EfficientNet-B0');
  const [testImage, setTestImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('');

  useEffect(() => {
    // Load initial test image
    loadRandomTestImage();
  }, []);

  const loadRandomTestImage = async () => {
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);
      
      const imageData = await apiService.getTestImage({ random: true });
      setTestImage(imageData);
    } catch (err) {
      setError('Failed to load test image');
      console.error('Error loading test image:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadCategoryTestImage = async () => {
    if (!selectedCategory) return;
    
    try {
      setLoading(true);
      setError(null);
      setPrediction(null);
      
      const imageData = await apiService.getTestImage({ 
        category: selectedCategory, 
        random: false 
      });
      setTestImage(imageData);
    } catch (err) {
      setError('Failed to load test image from selected category');
      console.error('Error loading category test image:', err);
    } finally {
      setLoading(false);
    }
  };

  const makePrediction = async () => {
    if (!testImage) return;
    
    try {
      setLoading(true);
      setError(null);
      
      const predictionRequest = {
        model_name: selectedModel,
        image_url: null,
        image_file: null
      };
      
      const result = await apiService.makePrediction(predictionRequest);
      setPrediction(result);
    } catch (err) {
      setError('Failed to make prediction');
      console.error('Error making prediction:', err);
    } finally {
      setLoading(false);
    }
  };

  const getImageUrl = (category, filename) => {
    return apiService.getImage(category, filename);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-success-600';
    if (confidence >= 0.6) return 'text-warning-600';
    return 'text-error-600';
  };

  const getConfidenceBgColor = (confidence) => {
    if (confidence >= 0.8) return 'bg-success-50';
    if (confidence >= 0.6) return 'bg-warning-50';
    return 'bg-error-50';
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-6xl mx-auto"
      >
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center">
            <ImageIcon className="w-8 h-8 mr-3 text-primary-500" />
            Test Image Classification
          </h1>
          <p className="text-gray-600">
            Test our AI models with random images from the aerial landscape dataset
          </p>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                {config.modelTypes.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>

            {/* Category Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Category (Optional)
              </label>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              >
                <option value="">Random Category</option>
                {config.categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </div>

            {/* Action Buttons */}
            <div className="flex items-end space-x-3">
              <button
                onClick={loadRandomTestImage}
                disabled={loading}
                className="flex-1 px-4 py-2 bg-primary-500 text-white rounded-md hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
              >
                <Shuffle className="w-4 h-4 mr-2" />
                Random Image
              </button>
              {selectedCategory && (
                <button
                  onClick={loadCategoryTestImage}
                  disabled={loading}
                  className="flex-1 px-4 py-2 bg-secondary-500 text-white rounded-md hover:bg-secondary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                >
                  <Target className="w-4 h-4 mr-2" />
                  Category Image
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Image Display */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <ImageIcon className="w-5 h-5 mr-2 text-primary-500" />
              Test Image
            </h2>
            
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full"
                />
              </div>
            ) : testImage ? (
              <div className="space-y-4">
                <div className="relative">
                  <img
                    src={getImageUrl(testImage.category, testImage.filename)}
                    alt={`Test image from ${testImage.category}`}
                    className="w-full h-64 object-cover rounded-lg border border-gray-200"
                  />
                  <div className="absolute top-2 left-2 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-sm">
                    {testImage.category}
                  </div>
                </div>
                
                <div className="text-center">
                  <p className="text-sm text-gray-600 mb-2">Image Category</p>
                  <p className="text-lg font-semibold text-gray-900">{testImage.category}</p>
                  <p className="text-sm text-gray-500">{testImage.filename}</p>
                </div>

                <button
                  onClick={makePrediction}
                  disabled={loading}
                  className="w-full px-4 py-3 bg-success-500 text-white rounded-lg hover:bg-success-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                >
                  <Brain className="w-5 h-5 mr-2" />
                  {loading ? 'Processing...' : 'Classify Image'}
                </button>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <div className="text-center">
                  <ImageIcon className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                  <p>No test image loaded</p>
                </div>
              </div>
            )}
          </motion.div>

          {/* Prediction Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
          >
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <Brain className="w-5 h-5 mr-2 text-primary-500" />
              Prediction Results
            </h2>
            
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <div className="text-center">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    className="w-8 h-8 border-4 border-primary-500 border-t-transparent rounded-full mx-auto mb-4"
                  />
                  <p className="text-gray-600">Analyzing image...</p>
                </div>
              </div>
            ) : prediction ? (
              <div className="space-y-6">
                {/* Main Prediction */}
                <div className="text-center p-6 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-4">
                    {prediction.predicted_class === testImage?.category ? (
                      <CheckCircle className="w-12 h-12 text-success-500" />
                    ) : (
                      <XCircle className="w-12 h-12 text-error-500" />
                    )}
                  </div>
                  
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">
                    {prediction.predicted_class}
                  </h3>
                  
                  <div className="flex items-center justify-center space-x-4 text-sm text-gray-600">
                    <span>Confidence: {prediction.confidence.toFixed(2)}%</span>
                    <span>•</span>
                    <span>Time: {prediction.processing_time.toFixed(2)}s</span>
                  </div>
                  
                  {prediction.predicted_class === testImage?.category && (
                    <div className="mt-3 inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-success-100 text-success-800">
                      <CheckCircle className="w-4 h-4 mr-1" />
                      Correct Prediction!
                    </div>
                  )}
                </div>

                {/* All Predictions */}
                <div>
                  <h4 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2 text-primary-500" />
                    All Classifications
                  </h4>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {Object.entries(prediction.all_predictions)
                      .sort(([,a], [,b]) => b - a)
                      .map(([category, confidence]) => (
                        <div
                          key={category}
                          className={`flex items-center justify-between p-3 rounded-lg ${
                            category === prediction.predicted_class 
                              ? 'bg-primary-50 border border-primary-200' 
                              : 'bg-gray-50'
                          }`}
                        >
                          <span className="font-medium text-gray-900">{category}</span>
                          <div className="flex items-center space-x-2">
                            <div className="w-20 bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  confidence >= 0.8 ? 'bg-success-500' :
                                  confidence >= 0.6 ? 'bg-warning-500' : 'bg-error-500'
                                }`}
                                style={{ width: `${confidence * 100}%` }}
                              />
                            </div>
                            <span className={`text-sm font-medium ${getConfidenceColor(confidence)}`}>
                              {(confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Model Info */}
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">Model Information</h4>
                  <div className="text-sm text-blue-800 space-y-1">
                    <p><strong>Model:</strong> {prediction.model_name}</p>
                    <p><strong>Processing Time:</strong> {prediction.processing_time.toFixed(3)} seconds</p>
                    <p><strong>Timestamp:</strong> {new Date(prediction.timestamp).toLocaleString()}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500">
                <div className="text-center">
                  <Brain className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                  <p>No prediction made yet</p>
                  <p className="text-sm">Load an image and click "Classify Image"</p>
                </div>
              </div>
            )}
          </motion.div>
        </div>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-8 bg-error-50 border border-error-200 rounded-lg p-4"
            >
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 text-error-500 mr-3" />
                <div>
                  <h3 className="text-sm font-medium text-error-800">Error</h3>
                  <p className="text-sm text-error-700 mt-1">{error}</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
};

export default TestImage;
