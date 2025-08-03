# Phone Price Predictor & Recommender Frontend

This is a modern Next.js frontend for the custom phone price predictor and recommender system.

## Features
- Custom phone builder form
- Price prediction using XGBoost backend
- 5 phone recommendations using k-NN backend
- Beautiful, responsive UI (Material-UI)
- Seamless integration with Flask backend

## Getting Started

### 1. Clone the repo and install dependencies
```sh
npx create-next-app@latest phone-predictor-frontend --typescript
cd phone-predictor-frontend
npm install @mui/material @emotion/react @emotion/styled @mui/icons-material axios react-hook-form
```

### 2. Add the provided code files (see below)
- Copy the generated `/pages`, `/components`, and `/utils` files into your Next.js project.

### 3. Run the frontend
```sh
npm run dev
```

### 4. Run the backend (Flask)
```sh
python app.py
```

### 5. Access the app
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend: [http://localhost:5000](http://localhost:5000)

---

## API Endpoints (used by frontend)
- `POST /predict` — Predicts price for custom phone
- `POST /recommend` — Returns 5 similar phones

---

## Customization
- Update the backend URL in `/utils/api.ts` if needed.
- Tweak UI in `/components` as desired.

---

## License
MIT 