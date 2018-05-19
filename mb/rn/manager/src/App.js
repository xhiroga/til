import React, { Component } from 'react';
import { View, Text } from 'react-native';
import { Provider } from 'react-redux';
import { createStore, applyMiddleware } from 'redux';
import firebase from 'firebase';
import ReduxThunk from 'redux-thunk'; // redux使いのためのミドルウェア
import reducers from './reducers';
import Router from './Router';

class App extends Component {
  componentWillMount(){
    const config = {
      apiKey: "AIzaSyBzMgxdGeD8qP3u9s8pet9lnAldYrDQuxY",
      authDomain: "manager-bd5de.firebaseapp.com",
      databaseURL: "https://manager-bd5de.firebaseio.com",
      projectId: "manager-bd5de",
      storageBucket: "manager-bd5de.appspot.com",
      messagingSenderId: "265457713134"
    };
    firebase.initializeApp(config);
  }

  render() {
    const store = createStore(reducers, {}, applyMiddleware(ReduxThunk))

    return(
      <Provider store={store}>
        <Router />
      </Provider>
    )
  }
}

export default App;
