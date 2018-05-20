import React, { Component } from 'react';
import { StyleSheet, Image, Text, TouchableOpacity, View } from 'react-native';
import Router from './src/Router';


// memo: import, from, class, extendsは全てES6の特徴
export default class App extends Component {
  render() {
    return (
      <Router />
    )

  }
}

class Greeting extends Component {
  render() {
    return (
      <Text>Hello, {this.props.name}</Text>
    );
  }
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
