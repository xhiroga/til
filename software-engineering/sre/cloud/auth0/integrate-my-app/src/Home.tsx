import React from 'react';
import './App.css';
import LoginButton from './LoginButton';
import LogoutButton from './LogoutButton';

function Home() {
  return (
    <div className='App'>
      <header className='App-header'>
        <a href={'/profile'}>
          <button>profile</button>
        </a>
        <LoginButton />
        <LogoutButton />
      </header>
    </div>
  );
}

export default Home;
