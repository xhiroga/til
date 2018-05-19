// appの中に表示される。
// connectはreact-redux側に干渉するためのツール

import React, { Component } from 'react';
import { ListView } from 'react-native';

import { connect } from 'react-redux';

import ListItem from './ListItem';

class LibraryList extends Component {
  componentWillMount() {
    console.log('foo');
    const ds = new ListView.DataSource({
      rowHasChanged: (r1,r2) => r1 !== r2
    });
    console.log('fuga')
    this.dataSource = ds.cloneWithRows(this.props.libraries);
    // 見せたいデータのラッパー。
  }

  // renderRowとして実行するとdataSourceを勝手に受け取ってくれるらしい
  renderRow(library) {
    return <ListItem library={library} />;
  }

  render() {
    console.log(this.props);
    return(
      <ListView
        dataSource = {this.dataSource}
        renderRow={this.renderRow}
      />
    );
  }
}

const mapStateToProps = state => {
  console.log('hoge');
  console.log(state);
  return {libraries:state.libraries} // これのkeyはこのmapStateToPropsが責任持つ
}; // takte global state propaty and aplicatikn props


export default connect(mapStateToProps)(LibraryList);
// when connect called, back another function, assgin LibraryList
