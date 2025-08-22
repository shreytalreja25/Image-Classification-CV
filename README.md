# Aerial Landscape Classification Dashboard

A comprehensive full-stack web application for AI-powered aerial landscape classification with real-time model testing, performance analytics, and WebSocket support.

## 🚀 Features

### Core Functionality
- **Real-time Image Classification**: Test AI models with random images from the dataset
- **Multi-Model Support**: EfficientNet-B0, MobileNetV2, ResNet-18, and Random Forest
- **Live Dashboard**: Real-time statistics and performance metrics
- **WebSocket Integration**: Real-time updates and training progress monitoring
- **Responsive Design**: Modern UI with Framer Motion animations

### Technical Features
- **FastAPI Backend**: High-performance Python API with async support
- **React Frontend**: Modern React 18 with hooks and functional components
- **MongoDB Atlas**: Cloud database for storing training stats and predictions
- **WebSocket Support**: Real-time bidirectional communication
- **Environment-based Configuration**: Separate local and production URLs
- **Docker Support**: Containerized deployment options

## 🏗️ Architecture

```
├── backend/                 # FastAPI backend
│   ├── api.py              # Main API endpoints
│   ├── config.py           # Configuration management
│   ├── database.py         # MongoDB integration
│   ├── ml_models.py        # ML model serving
│   ├── models.py           # Pydantic data models
│   └── main.py             # Application entry point
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API and WebSocket services
│   │   └── config/         # Frontend configuration
│   ├── public/             # Static assets
│   └── package.json        # Dependencies
└── docs/                   # Documentation
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **Motor**: Async MongoDB driver
- **PyMongo**: MongoDB Python driver
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **Pillow**: Image processing

### Frontend
- **React 18**: Modern React with hooks
- **Framer Motion**: Animation library
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Beautiful icon library
- **Recharts**: Charting library
- **Axios**: HTTP client

### Database
- **MongoDB Atlas**: Cloud-hosted MongoDB
- **Motor**: Async MongoDB operations

### Deployment
- **Render**: Backend hosting
- **Vercel**: Frontend hosting
- **MongoDB Atlas**: Database hosting

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB Atlas account
- Render account (for backend)
- Vercel account (for frontend)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/aerial-classification-dashboard.git
cd aerial-classification-dashboard
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your MongoDB URI and other settings

# Run the backend
python main.py
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp env.example .env
# Edit .env with your backend URL

# Run the frontend
npm start
```

### 4. Database Setup

1. Create a MongoDB Atlas cluster
2. Get your connection string
3. Update the `MONGODB_URI` in your backend `.env` file

## 🔧 Configuration

### Backend Environment Variables

```bash
# Development Mode
DEV_MODE=true

# Backend URLs
LOCAL_BACKEND_URL=http://localhost:8000
PRODUCTION_BACKEND_URL=https://your-backend.onrender.com

# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/aerial_classification
MONGODB_DB=aerial_classification

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Frontend Environment Variables

```bash
# Development Mode
REACT_APP_DEV_MODE=true

# Backend URLs
REACT_APP_LOCAL_BACKEND_URL=http://localhost:8000
REACT_APP_PRODUCTION_BACKEND_URL=https://your-backend.onrender.com

# WebSocket URLs
REACT_APP_LOCAL_WS_URL=ws://localhost:8000/ws
REACT_APP_PRODUCTION_WS_URL=wss://your-backend.onrender.com/ws
```

## 🚀 Deployment

### Backend Deployment (Render)

1. Push your code to GitHub
2. Connect your repository to Render
3. Create a new Web Service
4. Set environment variables
5. Deploy

### Frontend Deployment (Vercel)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Set environment variables
4. Deploy

### Environment Variables for Production

```bash
# Backend (.env on Render)
DEV_MODE=false
MONGODB_URI=your-mongodb-atlas-uri
JWT_SECRET_KEY=your-production-jwt-secret

# Frontend (.env on Vercel)
REACT_APP_DEV_MODE=false
REACT_APP_PRODUCTION_BACKEND_URL=https://your-backend.onrender.com
REACT_APP_PRODUCTION_WS_URL=wss://your-backend.onrender.com/ws
```

## 📊 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /models/{model_name}/info` - Get model information
- `POST /predict` - Make prediction
- `POST /test-image` - Get test image
- `GET /training-stats` - Get training statistics
- `GET /dashboard-stats` - Get dashboard statistics

### WebSocket Endpoints
- `WS /ws` - Real-time updates

## 🎯 Usage

### Testing Models
1. Navigate to the "Test Image" section
2. Select a model from the dropdown
3. Choose a random image or specific category
4. Click "Classify Image" to see results
5. View confidence scores and processing time

### Dashboard
1. View real-time statistics on the main dashboard
2. Monitor model performance metrics
3. Check recent predictions
4. Access quick actions for common tasks

### Model Management
1. View all available models
2. Check model status and performance
3. Access training configurations
4. Monitor training progress

## 🔍 Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   - Check your MongoDB Atlas connection string
   - Ensure your IP is whitelisted
   - Verify database user credentials

2. **Models Not Loading**
   - Check if model files exist in the models directory
   - Verify PyTorch installation
   - Check model file permissions

3. **WebSocket Connection Issues**
   - Verify WebSocket URL configuration
   - Check firewall settings
   - Ensure backend is running

4. **Frontend Build Errors**
   - Clear node_modules and reinstall
   - Check Node.js version compatibility
   - Verify environment variable configuration

### Debug Mode

Enable debug logging by setting:
```bash
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **COMP9517 Computer Vision Course** - Project specification and requirements
- **SkyView Aerial Landscape Dataset** - Training data
- **FastAPI Community** - Excellent documentation and examples
- **React Community** - Modern React patterns and best practices

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

## 🔮 Future Enhancements

- [ ] User authentication and authorization
- [ ] Model versioning and A/B testing
- [ ] Advanced analytics and reporting
- [ ] Model fine-tuning interface
- [ ] Batch prediction capabilities
- [ ] Export functionality for results
- [ ] Mobile app support
- [ ] Multi-language support
- [ ] Advanced visualization options
- [ ] Integration with cloud ML platforms

---

**Built with ❤️ for Computer Vision and AI enthusiasts**
