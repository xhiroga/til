import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import './App.css';
import Home from './Home';
import Profile from './Profile';

function App() {
  return (
    <Router>
      <Switch>
        <Route path={`/profile`}>
          <Profile />
        </Route>
        <Route path={`/`}>
          <Home />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
