import { combineReducers } from 'redux';
import LibraryReducer from './LibraryReducer';
import SelectionReducer from './SelectionReducer';

export default combineReducers({
  libraries: LibraryReducer,
  selectedLibraryId: SelectionReducer
});

// これが複数のactionに対応するように書く
// cosodateだったら検索結果の取得とか現在ページとか持ってる

// console.log(store.getState());
