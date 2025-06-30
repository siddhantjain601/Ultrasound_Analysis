import React, { useState } from 'react';
import Laptop from '../assets/laptop-nobg.png';

const Login = () => {
  const [form, setForm] = useState({ name: '', email: '', password: '' });
  const [message, setMessage] = useState('');

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name.toLowerCase()]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('');
    try {
      const response = await fetch('http://localhost:3000/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      if (response.ok) {
        setMessage('Signup successful! You can now log in.');
        setForm({ name: '', email: '', password: '' });
      } else {
        setMessage(data.msg || 'Signup failed.');
      }
    } catch (err) {
      setMessage('Server error. Please try again later.');
    }
  };

  return (
    <div className='relative w-full py-16 overflow-hidden'>
      <div className='max-w-[1240px] mx-auto grid md:grid-cols-2 items-center px-4'>
        <img src={Laptop} className='mx-auto w-80 md:w-full' alt="Laptop" />
        <div className="flex flex-col items-center">
          <h1 className='text-white text-4xl text-center mb-6'>Signup Here.</h1>
          {message && <div className="mb-4 text-center text-[#00df9a] font-semibold">{message}</div>}
          <form onSubmit={handleSubmit} className='w-full flex flex-col items-center'>
            <ul className='text-white flex flex-col items-center w-full gap-4'>
              <li className='w-full max-w-md'>
                <input
                  type='text'
                  name="Name"
                  placeholder="Name"
                  className="w-full h-12 text-lg p-3 rounded bg-gray-800 border border-gray-600 text-white"
                  value={form.name}
                  onChange={handleChange}
                  required
                />
              </li>
              <li className='w-full max-w-md'>
                <input
                  type='email'
                  name="Email"
                  placeholder="Email"
                  className="w-full h-12 text-lg p-3 rounded bg-gray-800 border border-gray-600 text-white"
                  value={form.email}
                  onChange={handleChange}
                  required
                />
              </li>
              <li className='w-full max-w-md pb-4'>
                <input
                  type='password'
                  name="Password"
                  placeholder="Password"
                  className="w-full h-12 text-lg p-3 rounded bg-gray-800 border border-gray-600 text-white"
                  value={form.password}
                  onChange={handleChange}
                  required
                />
              </li>
            </ul>
            <button
              type="submit"
              className='relative mx-auto my-4 w-[225px] h-9 bg-white text-black rounded cursor-pointer overflow-hidden group font-bold uppercase flex items-center justify-center'
            >
              <span className='absolute inset-0 w-0 group-hover:w-full h-full bg-[#00df9a] opacity-50 transition-all duration-300 ease-out'></span>
              <span className='relative z-10'>Click Here to Signup</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Login;
