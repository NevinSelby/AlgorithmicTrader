#!/bin/bash

echo "🚀 Setting up Trading Platform..."

# Backend setup
echo "📦 Installing backend dependencies..."
cd backend
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt

# Frontend setup
echo "📦 Installing frontend dependencies..."
cd ../frontend
npm install

echo "✅ Setup complete!"
echo ""
echo "To run the platform:"
echo "  1. Backend: cd backend && python main.py"
echo "  2. Frontend: cd frontend && npm run dev"
echo ""
echo "Then open http://localhost:3000 in your browser"

