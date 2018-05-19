import React from 'react';
import { View } from 'react-native';

// これをやる前に yarn add react-redux と yarn add reduxしておく
import { Provider } from 'react-redux';
import { createStore } from 'redux'; // reducersを渡してstoreを作ってくれる、いわばコンストラクタ

import reducers from './reducers';　// これはid:匿名関数の束になる
import LibraryList from './components/LibraryList';

import { Header } from './components/common';

const App = () => {
  return(
    <Provider store = {createStore(reducers)}>
      <View style={{flex:1}}>
        <Header headerText="Tech Stack" />
        <LibraryList />
      </View>
    </Provider>
  );
};

export default App;
