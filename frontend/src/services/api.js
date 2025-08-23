import axios from "axios";
import { API_BASE_URL } from "../config/api";

export const api = axios.create({
  baseURL: API_BASE_URL,
});

export const getDashboardStats = () => api.get("/dashboard-stats").then(r => r.data);
export const getTrainingStats = () => api.get("/training-stats").then(r => r.data);
export const getModels = () => api.get("/models").then(r => r.data);
export const getRandomTestImage = (payload) => api.post("/test-image", payload).then(r => r.data);
export const predict = (payload) => api.post("/predict", payload).then(r => r.data);

