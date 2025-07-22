import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

// In-memory storage for demo purposes
// In production, use a proper database
const users = [];

// Generate JWT token
const generateToken = (userId) => {
  return jwt.sign(
    { userId }, 
    process.env.JWT_SECRET || 'your-secret-key', 
    { expiresIn: '24h' }
  );
};

// Register user
export const register = async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // Validation
    if (!name || !email || !password) {
      return res.status(400).json({ 
        msg: 'Please provide name, email, and password' 
      });
    }

    // Check if user already exists
    const existingUser = users.find(user => user.email === email);
    if (existingUser) {
      return res.status(400).json({ 
        msg: 'User already exists with this email' 
      });
    }

    // Hash password
    const saltRounds = 10;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    // Create user
    const newUser = {
      id: users.length + 1,
      name,
      email,
      password: hashedPassword,
      createdAt: new Date().toISOString()
    };

    users.push(newUser);

    // Generate token
    const token = generateToken(newUser.id);

    res.status(201).json({
      msg: 'User registered successfully',
      token,
      user: {
        id: newUser.id,
        name: newUser.name,
        email: newUser.email
      }
    });

  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ msg: 'Server error during registration' });
  }
};

// Login user
export const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    // Validation
    if (!email || !password) {
      return res.status(400).json({ 
        msg: 'Please provide email and password' 
      });
    }

    // Find user
    const user = users.find(user => user.email === email);
    if (!user) {
      return res.status(400).json({ 
        msg: 'Invalid credentials' 
      });
    }

    // Check password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ 
        msg: 'Invalid credentials' 
      });
    }

    // Generate token
    const token = generateToken(user.id);

    res.json({
      msg: 'Login successful',
      token,
      user: {
        id: user.id,
        name: user.name,
        email: user.email
      }
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ msg: 'Server error during login' });
  }
};

// Get all users (for debugging - remove in production)
export const getUsers = (req, res) => {
  const safeUsers = users.map(user => ({
    id: user.id,
    name: user.name,
    email: user.email,
    createdAt: user.createdAt
  }));
  res.json(safeUsers);
};