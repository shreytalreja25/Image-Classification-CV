import axios from 'axios';
import config from '../config/config';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: config.backendUrl,
  timeout: config.defaults.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    // Handle common errors
    if (error.response?.status === 401) {
      // Unauthorized - redirect to login
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API service functions
export const apiService = {
  // Health check
  healthCheck: () => api.get('/health'),
  
  // Models
  getModels: () => api.get('/models'),
  getModelInfo: (modelName) => api.get(`/models/${modelName}/info`),
  
  // Predictions
  makePrediction: (predictionRequest) => api.post('/predict', predictionRequest),
  
  // Test images
  getTestImage: (testImageRequest) => api.post('/test-image', testImageRequest),
  
  // Statistics
  getTrainingStats: () => api.get('/training-stats'),
  getDashboardStats: () => api.get('/dashboard-stats'),
  
  // Images
  getImage: (category, filename) => `${config.backendUrl}/images/${category}/${filename}`,
};

// WebSocket service
export class WebSocketService {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = config.defaults.maxRetries;
    this.reconnectInterval = 5000;
    this.heartbeatInterval = null;
    this.listeners = new Map();
  }

  connect() {
    try {
      this.ws = new WebSocket(config.wsUrl);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.startHeartbeat();
        this.emit('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.stopHeartbeat();
        this.emit('disconnected');
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };

    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.stopHeartbeat();
  }

  scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, this.reconnectInterval * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
    }
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Send ping every 30 seconds
  }

  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  handleMessage(data) {
    const { type, ...payload } = data;
    
    switch (type) {
      case 'training_progress':
        this.emit('trainingProgress', payload.data);
        break;
      case 'prediction_update':
        this.emit('predictionUpdate', payload.data);
        break;
      case 'model_status_update':
        this.emit('modelStatusUpdate', payload.data);
        break;
      case 'system_notification':
        this.emit('systemNotification', payload.data);
        break;
      case 'heartbeat':
        this.emit('heartbeat', payload.data);
        break;
      case 'pong':
        // Heartbeat response
        break;
      default:
        console.log('Unknown WebSocket message type:', type);
    }
  }

  // Event listener methods
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in event listener:', error);
        }
      });
    }
  }

  // Send message to WebSocket
  send(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }
}

// Create singleton instance
export const wsService = new WebSocketService();

export default apiService;
