# 📈 Sales Forecasting App

A modern web application for **predicting future sales trends** with intuitive data visualization.  
Built with **Python**, **FastAPI**, **React**, and **Vue.js**, this app allows users to upload their own sales data or use demo datasets to generate accurate forecasts powered by **Facebook Prophet**.

---

## 🚀 Overview

The Sales Forecasting App provides a simple yet powerful interface for anyone who needs data-driven sales predictions — from small business owners to analysts and students.  
Users can visualize historical sales data, view trend breakdowns, and explore AI-generated forecasts in interactive charts.

---

## ✨ Features

- 📤 **Upload your own CSV** file with sales data  
- 📊 **Visualize** historical and forecasted data in dynamic charts  
- 🔮 **Forecast future sales** using **Prophet**, an industry-trusted time-series model  
- 🧰 **Try demo datasets** for quick exploration  
- ⚙️ **FastAPI backend** for model serving and API endpoints  
- 💻 **React + Vue.js frontend** for a responsive and interactive user experience  
- ☁️ **Deployed on Netlify (frontend)** and **Railway (backend)**  

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-------------|
| Frontend | React, Vue.js |
| Backend | FastAPI (Python) |
| Forecasting | Prophet |
| Deployment | Netlify (frontend), Railway (backend) |

---

## 🧩 Architecture

```text
User
 ├──> Frontend (React/Vue)
 │       ├── CSV upload
 │       ├── Visualization (charts)
 │       └── API requests
 └──> Backend (FastAPI)
         ├── Data preprocessing
         ├── Prophet forecasting
         └── REST API responses
