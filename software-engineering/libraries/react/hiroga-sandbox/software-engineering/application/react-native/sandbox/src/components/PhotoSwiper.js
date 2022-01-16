import React, { Component } from 'react';
import {
  Dimensions,
  StyleSheet,
  Image,
  Text,
  View
} from 'react-native';

import Swiper from 'react-native-swiper';

const window = Dimensions.get('window');

const styles = {
  container: {
    flex: 1
  },

  wrapper: {
  },

  slide: {
    flex: 1,
    justifyContent: 'center',
    backgroundColor: 'transparent'
  },

  image: {
    width: window.width,
    height: 240,
    flex: 1
  }
}

export default class PhotoSwiper extends Component {
  render() {
    return (
      <Swiper style={styles.wrapper} height={240}
        onMomentumScrollEnd={(e, state, context) => console.log('index:', state.index)}
        dot={<View style={{ backgroundColor: 'rgba(0,0,0,.2)', width: 5, height: 5, borderRadius: 4, marginLeft: 3, marginRight: 3, marginTop: 3, marginBottom: 3 }} />}
        activeDot={<View style={{ backgroundColor: '#000', width: 8, height: 8, borderRadius: 4, marginLeft: 3, marginRight: 3, marginTop: 3, marginBottom: 3 }} />}
        paginationStyle={{
          bottom: -23, left: null, right: 10
        }} loop>
        <View style={styles.slide} title={<Text numberOfLines={1}>Aussie tourist dies at Bali hotel</Text>}>
          <Image resizeMode='cover' style={styles.image} source={require('../assets/img/food_soba.jpg')} />
        </View>
        <View style={styles.slide} title={<Text numberOfLines={1}>Big lie behind Nine’s new show</Text>}>
          <Image resizeMode='cover' style={styles.image} source={require('../assets/img/food_mabo.jpg')} />
        </View>
        <View style={styles.slide} title={<Text numberOfLines={1}>Why Stone split from Garfield</Text>}>
          <Image resizeMode='cover' style={styles.image} source={require('../assets/img/food_ice.jpg')} />
        </View>
      </Swiper>
    );
  }
}