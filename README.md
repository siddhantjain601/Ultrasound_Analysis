```markdown
# Ultrasound-Analysis

A full-stack application for breast ultrasound image analysis using deep learning, featuring classification and segmentation with a modern React frontend and FastAPI backend.

---

## 📁 Project Structure

```
Ultrasound-Analysis/
├── backend/
│   ├── __pycache__/
│   ├── controllers/
│   │   └── authController.js
│   ├── ml/
│   │   ├── __pycache__/
│   │   ├── best_attention_unet_busi.pth
│   │   ├── ml_api.py
│   │   ├── resnet.pth
│   │   └── resnet.py
│   ├── models/
│   │   └── userModel.js
│   ├── node_modules/
│   ├── routes/
│   │   └── auth.js
│   ├── .env
│   ├── .gitignore
│   ├── db.js
│   ├── package-lock.json
│   ├── package.json
│   └── server.js
├── node_modules/
├── public/
│   └── vite.svg
├── src/
│   ├── assets/
│   ├── components/
│   │   ├── Dashboard.jsx
│   │   ├── Hero.jsx
│   │   ├── Login.jsx
│   │   ├── Navbar.jsx
│   │   └── Signup.jsx
│   ├── App.jsx
│   ├── index.css
│   ├── main.jsx
│   └── ...
├── .gitignore
├── eslint.config.js
├── index.html
├── package-lock.json
├── package.json
├── README.md
└── vite.config.js
```

---

## 🚀 Getting Started

### 1. Backend Setup

- **Python 3.8+** required.
- From the `backend/ml/` directory (or main backend dir if files are reorganized), install dependencies:
    ```
    pip install fastapi uvicorn pillow torch torchvision numpy scikit-learn python-multipart bcrypt psycopg2-binary supabase
    ```
- Place your trained model weights (`resnet.pth` and `best_attention_unet_busi.pth`) in the `ml/` folder.
- Start FastAPI server:
    ```
    uvicorn ml_api:app --reload --host 0.0.0.0 --port 8000
    ```
- The `/segment` endpoint will be available at [http://localhost:8000/segment](http://localhost:8000/segment).

### 2. Backend Node/Express Auth Service (Optional)

- Install dependencies in `backend/`:
    ```
    npm install
    ```
- Configure `.env` with your DB info (Supabase/Postgres).
- Start the Express server:
    ```
    node server.js
    ```
- The authentication routes will be available at [http://localhost:3000/api/auth/](http://localhost:3000/api/auth/).

### 3. Frontend Setup

- **Node.js 16+** recommended.
- Navigate to root (where `package.json` is found):
    ```
    npm install
    npm run dev
    ```
- The app runs by default at [http://localhost:5173](http://localhost:5173).

### 4. File Upload and Analysis Workflow

- Select an ultrasound image on the Dashboard page.
- **Preview:** Your image appears before analysis.
- Click "Analyze" to send the image to the backend ML pipeline.
- **Result:** Shows classification (`normal`, `benign`, `malignant`) and (if not `normal`) the segmentation mask.

---

## 💡 Key Features

- **Two-stage pipeline:**  
  - ResNet classifier predicts class.
  - Attention U-Net segments the region of interest if abnormal.

- **Live upload preview:**  
  Your image appears instantly upon selection.

- **Express/JWT authentication:**  
  Secure user sign-up/login using Node and Supabase/Postgres.

- **Tailwind-styled React frontend:**  
  Clean, responsive UI with navigation.

---

## ⚙️ Backend Dependencies

- fastapi
- uvicorn
- torch
- torchvision
- pillow
- numpy
- scikit-learn
- python-multipart
- bcrypt
- psycopg2-binary
- supabase

Install via:
```
pip install fastapi uvicorn torch torchvision pillow numpy scikit-learn python-multipart bcrypt psycopg2-binary supabase
```

---

## 🛠️ Frontend Dependencies

Specified in `package.json`; key libraries include:
- react, react-dom, react-router-dom  
- lucide-react (for icons)  
- tailwindcss  
- dotenv  
- axios  
- bcryptjs  
- classnames  

Install all with:
```
npm install
```

---

## 📝 Environment Configuration

Configure your database and secrets in `.env` for backend Express:
```
PORT=3000
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_supabase_endpoint
DB_PORT=5432
DB_DATABASE=your_db_name
JWT_SECRET=your_jwt_secret
```

---

## 🧩 Troubleshooting

| Problem                   | Action                                          |
|---------------------------|-------------------------------------------------|
| Port 8000/3000 in use     | Free it, or use a different port                |
| CORS errors               | Ensure FastAPI CORS middleware is enabled       |
| Auth not working          | Check DB connection, JWT secret, body parsing   |
| ML mask not shown         | Confirm correct image upload and file format    |
| FastAPI not responding    | Launch with `--host 0.0.0.0`                    |

---

## 🤝 Contributing

1. Fork the repo and create your branch.
2. Commit changes and open a PR.
3. For any issues, bug reports, or suggestions, use the issues tab.

---

## 📣 Notes

- For production, use proper SSL, JWT secret management, and CORS policies.
- Never commit private keys or credentials.

---

## 📧 Contact

For help or feedback, open an issue in the repository or reach out to the project maintainer.

---
```
Copy, edit, and paste this into your `README.md` at the project root for an out-of-the-box guide tailored to your folder structure and workflow.

[1] https://pplx-res.cloudinary.com/image/private/user_uploads/52545704/303a25ec-3c72-437c-9702-7d501d4b9393/image.jpg
[2] https://pplx-res.cloudinary.com/image/private/user_uploads/52545704/1bcef41e-74aa-4ca0-a56b-5d4904e97245/image.jpg
[3] https://pplx-res.cloudinary.com/image/private/user_uploads/52545704/0cae60ff-37d2-4b87-992c-ac954f95968c/image.jpg
[4] https://pplx-res.cloudinary.com/image/private/user_uploads/52545704/2a699bde-0c61-4199-9da5-87c735999486/image.jpg