import React from 'react';
import { ReactTyped } from 'react-typed';

const Hero = ({onLoginClick}) => {
    return(
        <div className='text-white'>
            <div className="opacity-50 absolute inset-0 -z-10 h-full w-full items-center px-5 py-24 [background:radial-gradient(125%_125%_at_50%_10%,#000_40%,#30cde6_100%)]"></div>
            <div className='max-w-[800px] mt-[-96px] w-full h-screen mx-auto text-center flex flex-col justify-center'>
                <p className='text-[#00df9a] font-bold text-xl'>Website for Ai For Fun.</p>
                <h1 className='md:text-7xl sm:text-6xl text-4xl font-bold'>Exploring Ai For Fun.</h1>
                <div className='p-2 justify-center items-center md:text-4xl sm:text-3xl text-xl flex whitespace-nowrap'>
                    This website was created by    
                    <ReactTyped className='md:text-4xl sm:text-3xl text-xl text-gray-400 ml-2'
                        strings={["Toshit", "Yash", "Pratyush"]}
                        typeSpeed={100}
                        backSpeed={70}
                        loop/> 
                </div>
                <button className='relative mx-auto my-4 w-[225px] h-9 bg-white text-black rounded cursor-pointer overflow-hidden group font-bold uppercase flex items-center justify-center' onClick={onLoginClick}>
                    <span className='absolute inset-0 w-0 group-hover:w-full h-full bg-[#00df9a] opacity-50 transition-all duration-300 ease-out'></span>
                    <span className='relative z-10'>Click Here to Signup</span>
                </button>
            </div>
        </div>
    )
}

export default Hero