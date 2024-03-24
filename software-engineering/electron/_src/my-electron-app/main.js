const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('node:path')

const createWindow = () => {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })

  win.loadFile('index.html')
}

// on('ready', ...) のラッパー
app.whenReady().then(() => {
  ipcMain.handle('ping', () => 'pong')
  createWindow()

  // アプリがreadyかつinactivateというのはmacOSのベストプラクティスを想定した挙動なので、
  // Windows向けのアプリを作りたい場合はデッドコードである。
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
