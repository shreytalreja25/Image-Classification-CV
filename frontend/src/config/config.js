const config = {
  // Development Mode
  devMode: process.env.REACT_APP_DEV_MODE === 'true',
  
  // Backend URLs
  backendUrl: process.env.REACT_APP_DEV_MODE === 'true' 
    ? process.env.REACT_APP_LOCAL_BACKEND_URL 
    : process.env.REACT_APP_PRODUCTION_BACKEND_URL,
  
  // Frontend URLs
  frontendUrl: process.env.REACT_APP_DEV_MODE === 'true' 
    ? process.env.REACT_APP_LOCAL_FRONTEND_URL 
    : process.env.REACT_APP_PRODUCTION_FRONTEND_URL,
  
  // WebSocket URLs
  wsUrl: process.env.REACT_APP_DEV_MODE === 'true' 
    ? process.env.REACT_APP_LOCAL_WS_URL 
    : process.env.REACT_APP_PRODUCTION_WS_URL,
  
  // App Configuration
  appName: process.env.REACT_APP_APP_NAME || 'Aerial Classification Dashboard',
  appVersion: process.env.REACT_APP_APP_VERSION || '1.0.0',
  appDescription: process.env.REACT_APP_APP_DESCRIPTION || 'AI-powered aerial landscape classification dashboard',
  
  // API Endpoints
  apiEndpoints: {
    health: '/health',
    models: '/models',
    modelInfo: '/models/{model_name}/info',
    predict: '/predict',
    testImage: '/test-image',
    trainingStats: '/training-stats',
    dashboardStats: '/dashboard-stats',
    images: '/images/{category}/{filename}',
  },
  
  // WebSocket Events
  wsEvents: {
    TRAINING_PROGRESS: 'training_progress',
    PREDICTION_UPDATE: 'prediction_update',
    MODEL_STATUS_UPDATE: 'model_status_update',
    SYSTEM_NOTIFICATION: 'system_notification',
    HEARTBEAT: 'heartbeat',
  },
  
  // Model Types
  modelTypes: [
    'EfficientNet-B0',
    'MobileNetV2', 
    'ResNet-18',
    'Random Forest'
  ],
  
  // Categories
  categories: [
    'Agriculture', 'Airport', 'Beach', 'City', 'Desert',
    'Forest', 'Grassland', 'Highway', 'Lake', 'Mountain',
    'Parking', 'Port', 'Railway', 'Residential', 'River'
  ],
  
  // Default values
  defaults: {
    pageSize: 10,
    refreshInterval: 30000, // 30 seconds
    maxRetries: 3,
    timeout: 10000, // 10 seconds
  }
};

export default config;
