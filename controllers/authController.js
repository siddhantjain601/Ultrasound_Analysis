import bcrypt from 'bcrypt';              // For password hashing(hpasswd in usermodel.js)
import jwt from 'jsonwebtoken';           // For token generation
import { createUser, findUserByEmail } from '../models/userModel.js';

export const register = async (req, res) => {
  try {
    const { name, email, password } = req.body;

    const existingUser = await findUserByEmail(email);
    if (existingUser) return res.status(400).json({ msg: 'User already exists' });
    //the above func returns error if a suer tries to regisster with email that exists

    const hpasswd = await bcrypt.hash(password, 10);
    const user = await createUser(name, email, hpasswd);
    res.status(201).json({ msg: 'User registered', user: { id: user.id, name: user.name, email: user.email } });
  } catch (error) {
    res.status(500).json({ msg: 'Server error', error: error.message }); //this try catch bock is to ensure if internal error is there with status 500
  }
};

export const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await findUserByEmail(email);
    if (!user) return res.status(400).json({ msg: 'Invalid credentials' });

    const passwdmatch = await bcrypt.compare(password, user.password);
    if (!passwdmatch) return res.status(400).json({ msg: 'Invalid credentials' });

    const token = jwt.sign({ id: user.id }, process.env.JWT_SECRET, { expiresIn: '1h' });

    res.json({
      msg: 'Login successful',
      token,
      user: { id: user.id, name: user.name, email: user.email },
    });
  } catch (error) {
    res.status(500).json({ msg: 'Server error', error: error.message });
  }
};
