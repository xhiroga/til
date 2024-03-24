const { contextBridge, ipcRenderer } = require('electron')

// レンダラー(MainWorld)に対する小窓を開ける(expose)って感じらしい
contextBridge.exposeInMainWorld('versions', {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron,
  // ipcMain.handle で定義した関数を呼ぶもの。画面キャプチャの結果もここからもらうのかな？
  ping: () => ipcRenderer.invoke('ping')
  // 関数だけでなく変数も公開できます
})