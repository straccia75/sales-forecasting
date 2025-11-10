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


## BACKEND SETUP

# Clone the repository
git clone https://github.com/straccia75/sales-forecasting.git
cd sales-forecasting/backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload

## FRONTEND SETUP

cd ../frontend

# Install dependencies
npm install

# Run development server
npm run dev


Your app will be available at:

Frontend: http://localhost:3000

Backend: http://localhost:8000


📊 Usage

Start both servers (frontend + backend).

Open the app in your browser.

Upload a CSV file (with date and sales columns) or choose a demo dataset.

Click “Forecast” to create predictions.

Explore interactive charts to analyze sales trends.

🧾 Example CSV Format (The system will detect a Date column no matter what the name is...)
date	sales
2023-01-01	1500
2023-01-02	1620
2023-01-03	1740

🌐 Deployment
Service	Platform
Frontend	Netlify

Backend	Railway

🪪 License $ Usage

This project is **proprietary** and not open source.  
All rights reserved © Luigi Straccia, 2025.

You may view the code, but **reproduction, modification, or redistribution** of any part of this project is **not permitted** without explicit written permission from the author.

👨‍💻 Author

Developed by Luigi Straccia
📧 straccia75@gmail.com

🌐 https://lmps.dev