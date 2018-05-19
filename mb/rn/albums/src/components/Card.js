import React from 'react';
import { View } from 'react-native';

const Card = (props) => {
  return (
    <View style ={styles.containerStyle}>
      {props.children}
    </View>
  );
};
// .childrenで親要素の内側で宣言されたもの全てにアクセス可能

const styles = {
  containerStyle:{
    borderWidth: 1,
    borderRadius: 2,
    borderColor: '#ddd',
    borderBottomWidth: 0,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 }, //影の伸びる方向
    shadowOpacity: 0.1, // 不透明度
    shadowRadius: 2,
    elevation: 1,
    marginLeft: 5,
    marginRight: 5,
    marginTop: 10
  }
}

export default Card;
