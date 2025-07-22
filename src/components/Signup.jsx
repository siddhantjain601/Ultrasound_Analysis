import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

const Signup = () => {
  const [form, setForm] = useState({ name: '', email: '', password: '', confirmPassword: '' });
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('');
    
    // Basic validation
    if (form.password !== form.confirmPassword) {
      setMessage('Passwords do not match.');
      return;
    }
    
    // Replace with your signup endpoint and logic
    try {
      const response = await fetch('http://localhost:3000/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: form.name,
          email: form.email,
          password: form.password
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setMessage('Signup successful! Redirecting to login...');
        // Redirect to login page after successful signup
        setTimeout(() => {
          navigate('/login');
        }, 2000);
      } else {
        setMessage(data.msg || 'Signup failed.');
      }
    } catch (err) {
      setMessage('Server error. Please try again later.');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#000300] py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-white">
            Create your account
          </h2>
          <p className="mt-2 text-center text-sm text-gray-400">
            Already have an account?{' '}
            <Link to="/login" className="font-medium text-[#00df9a] hover:text-[#00df9a]/80">
              Sign in
            </Link>
          </p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label htmlFor="name" className="sr-only">
                Full Name
              </label>
              <input
                id="name"
                name="name"
                type="text"
                required
                className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-600 placeholder-gray-400 text-white bg-gray-800 focus:outline-none focus:ring-[#00df9a] focus:border-[#00df9a] focus:z-10 sm:text-sm"
                placeholder="Full Name"
                value={form.name}
                onChange={handleChange}
              />
            </div>
            <div>
              <label htmlFor="email" className="sr-only">
                Email address
              </label>
              <input
                id="email"
                name="email"
                type="email"
                required
                className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-600 placeholder-gray-400 text-white bg-gray-800 focus:outline-none focus:ring-[#00df9a] focus:border-[#00df9a] focus:z-10 sm:text-sm"
                placeholder="Email address"
                value={form.email}
                onChange={handleChange}
              />
            </div>
            <div>
              <label htmlFor="password" className="sr-only">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-600 placeholder-gray-400 text-white bg-gray-800 focus:outline-none focus:ring-[#00df9a] focus:border-[#00df9a] focus:z-10 sm:text-sm"
                placeholder="Password"
                value={form.password}
                onChange={handleChange}
              />
            </div>
            <div>
              <label htmlFor="confirmPassword" className="sr-only">
                Confirm Password
              </label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                required
                className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-600 placeholder-gray-400 text-white bg-gray-800 focus:outline-none focus:ring-[#00df9a] focus:border-[#00df9a] focus:z-10 sm:text-sm"
                placeholder="Confirm Password"
                value={form.confirmPassword}
                onChange={handleChange}
              />
            </div>
          </div>

          <div>
            <button
              type="submit"
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-black bg-[#00df9a] hover:bg-[#00df9a]/90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#00df9a] focus:ring-offset-gray-800 transition-colors duration-200"
            >
              Sign up
            </button>
          </div>
        </form>
        {message && (
          <div className={`mt-4 p-3 rounded-md text-center ${
            message.includes('successful') 
              ? 'bg-green-900/50 text-green-400' 
              : 'bg-red-900/50 text-red-400'
          }`}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
};

export default Signup;