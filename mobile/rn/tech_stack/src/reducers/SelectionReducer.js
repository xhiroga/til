export default (state = null, action) => {
  // 引数でのこの書き方はデフォルトの値。
  switch (action.type) {
    case 'select_library':
      return action.payload;
    default:
      return state
  }
};
