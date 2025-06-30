import React, { useRef } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Login from './components/Login';

function App() {
  const loginRef = useRef(null);

  const scrollToLogin = () => {
    loginRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  return (
    <div>
      <Navbar onLoginClick={scrollToLogin}/>
      <Hero onLoginClick={scrollToLogin}/>
      <div ref={loginRef}>
        <Login onLoginClick={scrollToLogin}/>
      </div>
    </div>
  )
}

export default App
