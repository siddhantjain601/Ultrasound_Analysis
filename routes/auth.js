import express from 'express';
import { register, login } from '../controllers/authController.js';

const router = express.Router();

router.post('/register', register);  // POST /api/auth/register for thunderclient(in my vscode)
router.post('/login', login);        // POST /api/auth/login(same)

export default router;
