import './App.css'
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
} from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Models from './pages/Models'
import TestModel from './pages/TestModel'
import Predictions from './pages/Predictions'
import Training from './pages/Training'
import Settings from './pages/Settings'

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/models" element={<Models />} />
          <Route path="/test" element={<TestModel />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/training" element={<Training />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
