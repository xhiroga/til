import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

class Timer extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      date: new Date()
    }
  }

  componentDidMount() {
    this.timerID = setInterval(
      () => { this.tick() },
      1000
    )
  }

  componentWillUnmount() {
    clearInterval(this.timerID);
  }

  tick() {
    this.setState({
      date: new Date()
    })
  }

  render() {
    return (
      <Text>{this.state.date.toLocaleTimeString()}</Text>
    )
  }
}

export default class App extends React.Component {
  render() {
    return (
      <View style={styles.container}>
        <Timer />
      </View>
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
