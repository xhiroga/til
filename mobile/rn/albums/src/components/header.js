import React from 'react';
import { Text, View } from 'react-native';

// コンポーネント名はファイル名と揃えること
const Header = (props) => {
  const {textStyle, viewStyle} = styles;

  return (
    <View style={viewStyle}>
      <Text style={textStyle}>{props.headerText}</Text>
    </View>
  );
};

const styles = {
  viewStyle: {
    backgroundColor: '#F8F8F8',
    justifyContent: 'center',
    alignItems: 'center',
    height: 60,
    paddingTop: 15,
    shadowColor: '#000',
    shadowOffset: {witdh:0, height:2},
    shadowOpacity: 0.2,
    elevation: 2,
    position: 'relative'
  },
  textStyle:{
    fontSize: 20
  }
};

// exportステートメントで他の要素からアクセス可能にする。
// registerはルート要素のみ使用可能
export default Header;
