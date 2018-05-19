import React, {Component} from 'react';
import { StyleSheet, Image, Text, TouchableOpacity, View } from 'react-native';
import { Camera, Permissions } from 'expo';
// ES6のimport文は{}ありとなしの2通りがあり、importする対象によって書き方が変わる。
// {}なしのimportは、export defaultされたオブジェクトに対して用いる。それ以外は、{}のなかにメンバー名を指定してimportする。
// https://sbfl.net/blog/2017/07/26/es-modules-basics/

import { Blink } from './src/components';

// memo: import, from, class, extendsは全てES6の特徴
export default class App extends Component {
  state = {
    hasCameraPermission: null,
    type: Camera.Constants.Type.back,
  };

  async componentWillMount() {
    const { status } = await Permissions.askAsync(Permissions.CAMERA);
    this.setState({ hasCameraPermission: status === 'granted' });
  }

  render() {

    let pic = {
      url: 'https://1.bp.blogspot.com/-M9ZorsaCGzc/Vte24Cd6PLI/AAAAAAAA4T4/bIQPrZgCHho/s800/airgun_women_syufu.png'
    };

    // JSXの中では{}を使うことで変数を埋め込むことができる。
    // JSXタグの呼び出し時に渡したプロパティは、インスタンス変数props={}に格納される。
    const { hasCameraPermission } = this.state;
    if (hasCameraPermission === null) {
      return <View />;
    } else if (hasCameraPermission === false) {
      return <Text>No access to camera</Text>;
    } else {
      return (
        <View style={{ flex: 1 }}>
          <Greeting name="Hiro" />
          <Image source={pic} style={{width:386, height:220}}/>
          <Camera style={{ flex: 1 }} type={this.state.type}>
            <View
              style={{
                flex: 1,
                backgroundColor: 'transparent',
                flexDirection: 'row',
              }}>
              <TouchableOpacity
                style={{
                  flex: 0.1,
                  alignSelf: 'flex-end',
                  alignItems: 'center',
                }}
                onPress={() => {
                  this.setState({
                    type: this.state.type === Camera.Constants.Type.back
                      ? Camera.Constants.Type.front
                      : Camera.Constants.Type.back,
                  });
                }}>
                <Text
                  style={{ fontSize: 18, marginBottom: 10, color: 'white' }}>
                  {' '}Flip{' '}
                </Text>
              </TouchableOpacity>
            </View>
          </Camera>
        </View>
      );
    }
  }
}

class Greeting extends Component{
  render() {
    return (
      <Text>Hello, {this.props.name}</Text>
    );
  }
}

class Blink extends Component{

}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
