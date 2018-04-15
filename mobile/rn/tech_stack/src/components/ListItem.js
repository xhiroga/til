// ライフサイクルメソッドを使いたい時はclassコンポーネント

import React, {Component} from 'react';
import {
  Text,
  TouchableWithoutFeedback,
  View,
  LayoutAnimation
 } from 'react-native';
import { connect } from 'react-redux';

import { CardSection } from './common';
// import する名前が.jsファイルの名前なら、フォルダ指定だけで構わない

import * as actions from '../actions';
// * as means import by each keyword but one time

class ListItem extends Component {
  componentWillUpdate(){
    LayoutAnimation.spring();
  } // これだけ！！！


  renderDescription () {
    const {library, expanded} = this.props;

    if (expanded) {
      return (
        <CardSection>
          <Text style = {{flex:1}}>
            {library.description}
          </Text>
        </CardSection>
      );
    }
  }

  //クラスコンポーネント、コンストラクタで宣言しなくてもthis.props取れる
  render() {
    const { titleStyle } = styles;
    const { id, title } = this.props.library;

    return(
      <TouchableWithoutFeedback
        onPress={()=>this.props.selectLibrary(id)}
      >
        <View>
          <CardSection>
            <Text style = {titleStyle}>
              {title}
            </Text>
          </CardSection>
          {this.renderDescription()}
        </View>
      </TouchableWithoutFeedback>
    );
  }
}

const styles = {
  titleStyle :{
    fontSize: 18,
    paddingLeft: 15
  }
}

const mapStateToProps = (state,ownProps) => {
  const expanded = state.selectedLibraryId === ownProps.library.id;
  return { expanded: expanded};
  // 事前に計算しておくのが、こう言う簡単なアプリじゃない場合かなり役立つらしい
  // componentのレベルにしてあげるんやで
}

export default connect(mapStateToProps, actions)(ListItem);
// 1st, mapStateToProps, 2nd actionCreator


// 全体の流れ
// ボタンを押す > actionを呼ぶ > actionCreatorに渡される > reducerに渡る
// レンダリング再実施 > mapStatetoPropsにセットした関数が働く
// > コンストラクタが動き、レンダリング再実施

// つまり
// onXXXでセットし、connectで取得し、ライフサイクルメソッドで使う
