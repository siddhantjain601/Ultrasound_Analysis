import React, { useState } from 'react';
import { IoMenuSharp } from "react-icons/io5";
import { IoMdClose } from "react-icons/io";

const Navbar = ({onLoginClick}) => {
    const [nav, setNav] = useState(false);
    const handleNav = () =>{
        setNav(!nav)
    };
    const handleLoginClick = () => {
        onLoginClick();       // Scroll to Login section
        setNav(false);        // Close mobile menu if open
    };

    return(
        <div className='text-white flex justify-between items-center h-24 max-w-[1240px] mx-auto px-4'>
            <h1 className='w-full md:text-4xl text-3xl font-bold text-[#00df9a]'>Ai For Fun.</h1>
            <ul className='uppercase font-semibold hidden md:flex'>
                <li className='p-4 cursor-pointer whitespace-nowrap'>Home</li>
                <li className='p-4 cursor-pointer whitespace-nowrap'>Contacts</li>
                <li className='relative px-4 py-2 mt-2 h-9 ml-4 bg-white text-black rounded cursor-pointer whitespace-nowrap group' onClick={handleLoginClick}>
                    <span className='opacity-50 brightness-80 absolute w-0 group-hover:w-full transition-all ease-out duration-300 rounded h-9 -mt-2 -ml-4 bg-[#00df9a]'></span>
                    <span className='relative z-10 transition-colors duration-300'>Login</span>
                </li>
            </ul>
            <div onClick={handleNav} className='cursor-pointer block md:hidden'>
                {nav ? <IoMdClose size={25}/> : <IoMenuSharp size={25}/>}
            </div>
            <div className={`fixed top-0 left-0 w-[60%] h-full bg-[#000300] border-r border-r-gray-900 z-50 transform ${nav ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 ease-in-out block md:hidden`}>
            <h1 className='w-full text-3xl font-bold text-[#00df9a] m-4'>Ai For Fun.</h1>
            <ul className='uppercase font-semibold p-4'>
                <li className='py-4 cursor-pointer whitespace-nowrap'>Home</li>
                <li className='py-4 cursor-pointer whitespace-nowrap'>Contacts</li>
                <li className='py-4 cursor-pointer whitespace-nowrap' onClick={handleLoginClick}>Login</li>
            </ul>
            </div>
        </div>
    )
}

export default Navbar