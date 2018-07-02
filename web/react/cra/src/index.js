import React from 'react';
import ReactDOM from 'react-dom';
import createStore from 'redux';
import './index.css';
import App from './App';
import registerServiceWorker from './registerServiceWorker';

// Reduxの導入について。
// Action, ActionObject, Reducerを実装する。

ReactDOM.render(<App />, document.getElementById('root'));
registerServiceWorker();