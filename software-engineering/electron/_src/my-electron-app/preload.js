const { contextBridge } = require('electron')

// レンダラー(MainWorld)に対する小窓を開ける(expose)って感じらしい
contextBridge.exposeInMainWorld('versions', {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron
  // 関数だけでなく変数も公開できます
})