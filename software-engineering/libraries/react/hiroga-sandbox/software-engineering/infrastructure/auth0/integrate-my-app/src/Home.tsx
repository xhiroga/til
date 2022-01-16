import React from 'react';
import './App.css';
import GetPermissionButton from './GetPermissionButton';
import LoginButton from './LoginButton';
import LogoutButton from './LogoutButton';

function Home() {
  return (
    <div className='App'>
      <header className='App-header'>
        <LoginButton />
        <a href={'/profile'}>
          <button>profile</button>
        </a>
        <GetPermissionButton />
        <LogoutButton />
      </header>
    </div>
  );
}

export default Home;
